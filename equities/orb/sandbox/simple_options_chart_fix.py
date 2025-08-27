# Simple fix to add options P&L to your existing chart
# Just replace the annotation section with this code:

# After calculating stock pnl_pct, add this before the annotation:

# Calculate options P&L
strike_price = round(signal['entry_price'])
iv = 0.25  # Base IV
entry_time_to_expiry = 6/365  # ~6 hours to market close
exit_time_to_expiry = 2/365   # ~2 hours to market close

if signal['type'] == 'LONG':
    entry_option_price = options_pricer.black_scholes_call(
        signal['entry_price'], strike_price, entry_time_to_expiry, 0.05, iv
    )
    exit_option_price = options_pricer.black_scholes_call(
        exit_price, strike_price, exit_time_to_expiry, 0.05, iv
    )
else:
    entry_option_price = options_pricer.black_scholes_put(
        signal['entry_price'], strike_price, entry_time_to_expiry, 0.05, iv
    )
    exit_option_price = options_pricer.black_scholes_put(
        exit_price, strike_price, exit_time_to_expiry, 0.05, iv
    )

contracts = max(1, int(1000 / (entry_option_price * 100)))
options_pnl = (exit_option_price - entry_option_price) * 100 * contracts
options_roi = (options_pnl / 1000) * 100

# Then change the annotation to:
ax.annotate(f'1M {signal["type"]} ENTRY\n${signal["entry_price"]:.2f}\nSTOCK: {pnl_pct:+.1f}%\nOPTIONS: {options_roi:+.0f}%\nHIGH ({confidence_score:.1f})', 
           xy=(signal['time'], signal['entry_price']),
           xytext=(30, 50), textcoords='offset points',
           fontsize=11, color=entry_color, weight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                    alpha=0.9, edgecolor=entry_color, linewidth=2),
           arrowprops=dict(arrowstyle='->', color=entry_color, lw=2))

# And change the exit annotation to:
ax.annotate(f'{exit_reason} EXIT (WICK)\n${exit_price:.2f}\nSTOCK: {pnl_pct:+.1f}%\nOPTIONS: {options_roi:+.0f}%', 
           xy=(exit_time, exit_price),
           xytext=(30, -50), textcoords='offset points',
           fontsize=11, color=exit_color, weight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                    alpha=0.9, edgecolor=exit_color, linewidth=2),
           arrowprops=dict(arrowstyle='->', color=exit_color, lw=2))