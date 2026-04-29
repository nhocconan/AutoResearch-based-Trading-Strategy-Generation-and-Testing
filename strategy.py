#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter (price > 1w EMA50) and volume confirmation
# In bull markets: long when price breaks above Donchian(20) high + price > 1w EMA50 + volume spike
# In bear markets: short when price breaks below Donchian(20) low + price < 1w EMA50 + volume spike
# Uses weekly EMA50 to ensure we only trade with the major trend, avoiding counter-trend whipsaws
# Volume confirmation (>2.0x 20-period average) ensures institutional participation
# Designed for ~12-25 trades/year on 6h timeframe to minimize fee drag

name = "6h_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation (on 6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Handle exits: reverse position when opposite breakout occurs
        if position == 1:  # Long position
            # Exit long: price breaks below Donchian low (20-period)
            if curr_low < curr_lowest_low:
                signals[i] = -0.25  # Reverse to short
                position = -1
            else:
                signals[i] = 0.25   # Maintain long
                
        elif position == -1:  # Short position
            # Exit short: price breaks above Donchian high (20-period)
            if curr_high > curr_highest_high:
                signals[i] = 0.25   # Reverse to long
                position = 1
            else:
                signals[i] = -0.25  # Maintain short
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above Donchian high in uptrend (price > 1w EMA50)
            if vol_confirm and curr_close > curr_ema50_1w:
                if curr_high > curr_highest_high:
                    signals[i] = 0.25
                    position = 1
            # Short entry: price breaks below Donchian low in downtrend (price < 1w EMA50)
            elif vol_confirm and curr_close < curr_ema50_1w:
                if curr_low < curr_lowest_low:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
    
    return signals