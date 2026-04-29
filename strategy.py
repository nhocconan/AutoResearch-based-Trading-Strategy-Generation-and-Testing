#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA200 trend filter + volume confirmation
# Long when price breaks above Donchian upper channel, close > 1w EMA200, and volume > 1.5x 20-period average
# Short when price breaks below Donchian lower channel, close < 1w EMA200, and volume > 1.5x 20-period average
# Exit when price touches opposite Donchian channel or trend EMA crossover
# Uses discrete position sizing (0.30) to balance capture and risk.
# Donchian channels provide clear breakout levels, 1w EMA200 filters for higher-timeframe trend alignment.
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to avoid overtrading.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend.

name = "1d_Donchian20_1wEMA200_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation: volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * avg_volume_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Donchian and 1w EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_threshold[i])):
            signals[i] = 0.0
            continue
        
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_ema200_1w = ema_200_1w_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_volume = volume[i]
        curr_volume_threshold = volume_threshold[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price touches Donchian lower channel OR trend EMA crossover (close < 1w EMA200)
            if curr_low <= curr_lowest_20 or curr_close < curr_ema200_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price touches Donchian upper channel OR trend EMA crossover (close > 1w EMA200)
            if curr_high >= curr_highest_20 or curr_close > curr_ema200_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper channel, close > 1w EMA200, and volume confirmation
            if curr_high > curr_highest_20 and curr_close > curr_ema200_1w and curr_volume > curr_volume_threshold:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below Donchian lower channel, close < 1w EMA200, and volume confirmation
            elif curr_low < curr_lowest_20 and curr_close < curr_ema200_1w and curr_volume > curr_volume_threshold:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals