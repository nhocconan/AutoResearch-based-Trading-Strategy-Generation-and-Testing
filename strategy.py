#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly ATR for volatility measurement
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Weekly ATR ratio (current ATR / 20-period average) for volatility regime
    atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1w / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Daily ATR for position sizing and stop
    tr1_d = np.abs(high - low)
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr2_d[0] = tr3_d[0] = 0
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade in low volatility (ATR ratio < 0.8)
        low_volatility = atr_ratio_aligned[i] < 0.8
        
        # Long conditions: price breaks above upper Donchian + low volatility + volume
        long_breakout = (close[i] > highest_high[i-1] and low_volatility and volume_filter[i])
        # Short conditions: price breaks below lower Donchian + low volatility + volume
        short_breakout = (close[i] < lowest_low[i-1] and low_volatility and volume_filter[i])
        
        if long_breakout:
            signals[i] = 0.30
            position = 1
        elif short_breakout:
            signals[i] = -0.30
            position = -1
        # Exit conditions: opposite Donchian breakout
        elif position == 1 and close[i] < lowest_low[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_VolatilityFilter_Donchian20_Breakout_1wATR"
timeframe = "1d"
leverage = 1.0