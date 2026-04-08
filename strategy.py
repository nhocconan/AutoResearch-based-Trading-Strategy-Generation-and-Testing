# 4h_triple_confirmation_breakout_v1
# Hypothesis: Combines 4h price action (breaking above/below 20-period high/low) with 1d trend confirmation (price above/below 200 EMA) and volume surge (2x average volume).
# Trades only when all three conditions align, reducing false signals.
# Long when: price breaks above 20-period high, price > 200 EMA on 1d, volume > 2x average.
# Short when: price breaks below 20-period low, price < 200 EMA on 1d, volume > 2x average.
# Exit when price breaks back below the 20-period high (for longs) or above the 20-period low (for shorts).
# Uses strict entry conditions to limit trades to 20-40 per year per symbol, avoiding fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_triple_confirmation_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h price channel: 20-period high/low for breakout
    period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=period, min_periods=period).max().values
    lowest_low = low_series.rolling(window=period, min_periods=period).min().values
    
    # Volume filter: 2x 20-period average
    vol_ma_period = 20
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values
    vol_surge = volume > 2 * vol_ma
    
    # Get 1d data for trend confirmation (price vs 200 EMA)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate 200 EMA on 1d
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Align to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(period, vol_ma_period, 200) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks back below 20-period high
            if close[i] < highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks back above 20-period low
            if close[i] > lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above 20-period high, price > 200 EMA on 1d, volume surge
            if (close[i] > highest_high[i] and 
                close[i] > ema200_1d_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below 20-period low, price < 200 EMA on 1d, volume surge
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals