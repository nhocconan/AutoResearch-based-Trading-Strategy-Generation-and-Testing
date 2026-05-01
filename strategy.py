#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and 12h EMA50 trend filter.
# Long when price breaks above upper Donchian AND price > 12h EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below lower Donchian AND price < 12h EMA50 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years.
# Donchian provides clear structure, volume confirms momentum, 12h EMA50 filters counter-trend whipsaws.

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 12h trend: price > EMA50 = uptrend, price < EMA50 = downtrend
    uptrend_12h = close > ema_50_aligned  # Will be aligned inside loop via prices index
    downtrend_12h = close < ema_50_aligned
    
    # Donchian channels (20-period) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 4h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > highest_high[i]  # break above upper Donchian
        breakout_down = curr_low < lowest_low[i]   # break below lower Donchian
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper Donchian AND uptrend AND volume confirmation
            if (breakout_up and 
                uptrend_12h[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower Donchian AND downtrend AND volume confirmation
            elif (breakout_down and 
                  downtrend_12h[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower Donchian (stoploss) OR trend changes to downtrend
            if (curr_low < lowest_low[i] or 
                not uptrend_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian (stoploss) OR trend changes to uptrend
            if (curr_high > highest_high[i] or 
                not downtrend_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals