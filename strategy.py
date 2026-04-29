#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume spike
# Long when price breaks above Donchian(20) high, close > 1d EMA34, and volume > 1.5x 20-period avg volume
# Short when price breaks below Donchian(20) low, close < 1d EMA34, and volume > 1.5x 20-period avg volume
# Exit when price touches opposite Donchian(20) level or trend EMA crossover
# Uses discrete position sizing (0.25) to balance capture and risk.
# Donchian channels provide clear breakout levels, EMA34 filters trend direction, volume confirms conviction.
# 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to avoid overtrading.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend.

name = "12h_Donchian20_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate volume spike filter: volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 34, 20)  # Donchian, 1d EMA34, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price touches Donchian low OR trend EMA crossover (close < 1d EMA34)
            if curr_low <= curr_lowest_low or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches Donchian high OR trend EMA crossover (close > 1d EMA34)
            if curr_high >= curr_highest_high or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high, close > 1d EMA34, and volume spike
            if curr_high > curr_highest_high and curr_close > curr_ema34_1d and curr_volume_spike:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low, close < 1d EMA34, and volume spike
            elif curr_low < curr_lowest_low and curr_close < curr_ema34_1d and curr_volume_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals