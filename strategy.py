#!/usr/bin/env python3
"""
6h_WeeklyCamarilla_R4_S4_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop
Hypothesis: Weekly Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation captures strong momentum moves while avoiding false breakouts. Designed for low trade frequency (<30/year) to minimize fee drag in 6h timeframe. Works in both bull (breakout continuation) and bear (strong downtrend breaks) markets by trading only institutional-grade levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels (R4/S4 are strongest levels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for weekly lookback
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly Camarilla R4 and S4 levels (using previous weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_prev = df_1w['close'].values
    
    # Camarilla R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    R4 = close_1w_prev + (high_1w - low_1w) * 1.1 / 2
    S4 = close_1w_prev - (high_1w - low_1w) * 1.1 / 2
    
    # Align weekly levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Volume confirmation: 2.5x average volume (very strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # ATR for stoploss (using 20-period ATR for 6h timeframe)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of weekly lookback (implicit), 1d EMA (50), volume MA (50), ATR (20)
    start_idx = max(50, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        R4_val = R4_aligned[i]
        S4_val = S4_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above weekly R4 with volume confirmation and uptrend
            long_signal = (high_val > R4_val) and (volume_val > 2.5 * vol_ma_val) and (close_val > ema_50_1d_val)
            # Short: price breaks below weekly S4 with volume confirmation and downtrend
            short_signal = (low_val < S4_val) and (volume_val > 2.5 * vol_ma_val) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ATR stoploss or trend reversal
            if (close_val < entry_price - 3.0 * atr_val or 
                close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR stoploss or trend reversal
            if (close_val > entry_price + 3.0 * atr_val or 
                close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyCamarilla_R4_S4_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0