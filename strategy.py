# 1. Hypothesis: Price tends to revert to the volume-weighted average price (VWAP) over a day.
#   - Long when price is significantly below the daily VWAP and shows signs of reversal (RSI < 30).
#   - Short when price is significantly above the daily VWAP and shows signs of reversal (RSI > 70).
#   - The VWAP is calculated on the 1-day chart and aligned to the 6h timeframe to avoid look-ahead.
#   - This mean-reversion strategy works in both bull and bear markets as it fades extremes.
#   - Target: 15-30 trades per year (60-120 over 4 years).

# 2. Implementation:
#   - Calculate VWAP on the 1-day timeframe using typical price and volume.
#   - Align the daily VWAP to the 6h chart.
#   - Use a 14-period RSI on the 6h chart for entry timing.
#   - Enter long when 6h price < daily VWAP and RSI < 30.
#   - Enter short when 6h price > daily VWAP and RSI > 70.
#   - Exit when price crosses back to the VWAP or RSI returns to neutral territory (40-60).
#   - Position size is fixed at 0.25 to manage risk.

# 3. Risk Management:
#   - Fixed position size of 0.25 limits drawdown.
#   - Exit conditions are based on mean reversion to VWAP, acting as a dynamic take-profit.
#   - No additional stop-loss is used, relying on the mean-reversion logic to close losing trades
#     when price moves back towards VWAP, which is a form of time-based exit.

#!/usr/bin/env python3
"""
6h_vwap_mean_reversion_1d_rsi_v1
Hypothesis: Mean reversion to daily VWAP with RSI filter for entry timing.
Works in bull/bear markets by fading deviations from the day's fair value.
Target: 15-30 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vwap_mean_reversion_1d_rsi_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate VWAP on 1d: sum(price * volume) / sum(volume)
    # Using typical price for VWAP calculation
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align 1d VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back to VWAP or RSI returns to neutral (>40)
            if close[i] >= vwap_1d_aligned[i] or rsi[i] > 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses back to VWAP or RSI returns to neutral (<60)
            if close[i] <= vwap_1d_aligned[i] or rsi[i] < 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price below VWAP and RSI oversold (<30)
            if close[i] < vwap_1d_aligned[i] and rsi[i] < 30:
                position = 1
                signals[i] = 0.25
            # Short entry: price above VWAP and RSI overbought (>70)
            elif close[i] > vwap_1d_aligned[i] and rsi[i] > 70:
                position = -1
                signals[i] = -0.25
    
    return signals