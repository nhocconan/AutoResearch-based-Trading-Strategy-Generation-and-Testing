#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian breakout with 12h volatility regime and volume confirmation.
Long when price breaks above 6h Donchian high (20) during low volatility (12h ATR < 12h ATR MA) with volume surge.
Short when price breaks below 6h Donchian low (20) during low volatility with volume surge.
Exits when price crosses 6h EMA(20) in opposite direction or volatility expands.
Designed for low turnover: ~15-30 trades/year per symbol.
"""
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
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h ATR(14) for volatility regime
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_12h = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    
    # 6h Donchian channels (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h EMA(20) for exit
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # 12h index (2 bars per day: 24/12 = 2)
        idx_12h = i // 2
        if idx_12h < 20:  # need enough for ATR MA
            continue
        
        # Get previous 12h ATR and MA to avoid look-ahead
        atr_prev = atr_12h[idx_12h - 1] if idx_12h - 1 < len(atr_12h) else atr_12h[-1]
        atr_ma_prev = atr_ma_12h[idx_12h - 1] if idx_12h - 1 < len(atr_ma_12h) else atr_ma_12h[-1]
        if np.isnan(atr_prev) or np.isnan(atr_ma_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        atr_arr = np.full(len(df_12h), atr_prev)
        atr_ma_arr = np.full(len(df_12h), atr_ma_prev)
        atr_12h_6h = align_htf_to_ltf(prices, df_12h, atr_arr)[i]
        atr_ma_12h_6h = align_htf_to_ltf(prices, df_12h, atr_ma_arr)[i]
        
        if position == 0:
            # Long: Donchian breakout + low volatility + volume surge
            if (close[i] > donch_high[i] and 
                atr_12h_6h < atr_ma_12h_6h and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: Donchian breakdown + low volatility + volume surge
            elif (close[i] < donch_low[i] and 
                  atr_12h_6h < atr_ma_12h_6h and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price below EMA(20) or volatility expansion
            if close[i] < ema_20[i] or atr_12h_6h > atr_ma_12h_6h * 1.5:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price above EMA(20) or volatility expansion
            if close[i] > ema_20[i] or atr_12h_6h > atr_ma_12h_6h * 1.5:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian_VolRegime_Volume"
timeframe = "6h"
leverage = 1.0