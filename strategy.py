#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Donchian breakouts capture momentum in trending markets. 1d EMA34 ensures we only trade
# in direction of higher timeframe trend. Volume spike (>2x 20-period average) confirms
# institutional participation. Designed for 12-25 trades/year on 12h timeframe to minimize
# fee drag while maintaining edge in both bull and bear markets via trend filter.

name = "12h_Donchian20_1dEMA34_VolumeSpike_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for spike confirmation (on 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        
        # Handle exits: reverse position when opposite breakout occurs
        if position == 1:  # Long position
            # Exit long when price breaks below lower Donchian channel
            if curr_low < curr_lower:
                signals[i] = -0.30  # Reverse to short
                position = -1
            else:
                signals[i] = 0.30   # Maintain long
                
        elif position == -1:  # Short position
            # Exit short when price breaks above upper Donchian channel
            if curr_high > curr_upper:
                signals[i] = 0.30   # Reverse to long
                position = 1
            else:
                signals[i] = -0.30  # Maintain short
                
        else:  # Flat - look for new breakout entries with trend and volume filters
            # Volume confirmation: current volume > 2.0x 20-period average (strict to reduce trades)
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above upper Donchian + uptrend (close > 1d EMA34) + volume
            if vol_confirm and curr_close > curr_ema34_1d:
                if curr_high > curr_upper:  # Breakout confirmation
                    signals[i] = 0.30
                    position = 1
            # Short entry: price breaks below lower Donchian + downtrend (close < 1d EMA34) + volume
            elif vol_confirm and curr_close < curr_ema34_1d:
                if curr_low < curr_lower:  # Breakdown confirmation
                    signals[i] = -0.30
                    position = -1
            else:
                signals[i] = 0.0
    
    return signals