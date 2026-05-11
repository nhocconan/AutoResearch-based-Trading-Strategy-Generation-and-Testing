# Based on extensive testing, I'm focusing on a 4h strategy with strong entry conditions to limit trades.
# The hypothesis: A 4h Donchian breakout combined with 1d trend filter and volume confirmation
# will generate ~30-50 trades/year with sufficient edge in both bull and bear markets.
# The 1d trend filter adapts to market regime, while volume confirmation ensures breakout strength.
# Entry/exit logic is designed to be simple yet effective, avoiding overtrading.

#!/usr/bin/env python3
name = "4h_Donchian_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = ema50_1d > np.roll(ema50_1d, 1)  # Rising EMA50
    trend_down_1d = ema50_1d < np.roll(ema50_1d, 1)  # Falling EMA50
    
    # Calculate 4h Donchian channels (20-period)
    donchian_len = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(donchian_len - 1, n):
        highest_high[i] = np.max(high[i-donchian_len+1:i+1])
        lowest_low[i] = np.min(low[i-donchian_len+1:i+1])
    
    # Calculate 20-period volume average for confirmation
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    # For early periods, use available data
    for i in range(1, 20):
        vol_ma20[i] = np.mean(volume[:i+1])
    
    # Align 1d trend to 4h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band in uptrend with volume surge
            if (close[i] > highest_high[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band in downtrend with volume surge
            elif (close[i] < lowest_low[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Donchian lower band or trend turns down
            if (close[i] < lowest_low[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Donchian upper band or trend turns up
            if (close[i] > highest_high[i] or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals