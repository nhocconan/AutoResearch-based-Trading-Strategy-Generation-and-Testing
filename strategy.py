#!/usr/bin/env python3
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
    
    # Get 1d data for higher timeframe context (as per instructions: Primary=1d, HTF=1w)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for even higher timeframe trend (HTF=1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA 34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w EMA 34 for higher timeframe trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d Donchian channels (20-period for structure on daily timeframe)
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to the current timeframe
    highest_high_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_high_1d)
    lowest_low_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_1d)
    
    # Volume filter: volume > 1.8x 30-period average (strong filter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high_1d_aligned[i]) or np.isnan(lowest_low_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/both EMAs for long, below both for short
        price_above_both_emas = close[i] > ema_34_1d_aligned[i] and close[i] > ema_34_1w_aligned[i]
        price_below_both_emas = close[i] < ema_34_1d_aligned[i] and close[i] < ema_34_1w_aligned[i]
        
        # Long conditions: price breaks above 1d Donchian high + above both EMAs + volume
        long_breakout = (close[i] > highest_high_1d_aligned[i-1] and price_above_both_emas and volume_filter[i])
        # Short conditions: price breaks below 1d Donchian low + below both EMAs + volume
        short_breakout = (close[i] < lowest_low_1d_aligned[i-1] and price_below_both_emas and volume_filter[i])
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite 1d Donchian breakout
        elif position == 1 and close[i] < lowest_low_1d_aligned[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high_1d_aligned[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_1dEMA34_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0