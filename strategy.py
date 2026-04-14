#!/usr/bin/env python3
"""
Hypothesis: 12-hour timeframe strategy using 1-day Donchian channel breakouts
combined with 1-week volatility regime and volume confirmation.
Long when price breaks above 1-day Donchian high (20) during low weekly volatility
with volume surge. Short when price breaks below 1-day Donchian low (20) during
low weekly volatility with volume surge. Exits when price crosses 1-day EMA(20)
in opposite direction or volatility expands significantly.
Designed for low turnover: ~15-30 trades/year per symbol to minimize fee drag.
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
    
    # Load 1-day data once for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day Donchian channels (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1-day EMA(20) for exit
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load 1-week data for volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1-week ATR(14) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # 1-day index (2 bars per day: 24/12 = 2)
        idx_1d = i // 2
        if idx_1d < 20:  # need enough for Donchian calculation
            continue
        
        # Get previous 1-day values to avoid look-ahead
        donch_high_prev = donch_high[idx_1d - 1] if idx_1d - 1 < len(donch_high) else donch_high[-1]
        donch_low_prev = donch_low[idx_1d - 1] if idx_1d - 1 < len(donch_low) else donch_low[-1]
        ema_20_prev = ema_20_1d[idx_1d - 1] if idx_1d - 1 < len(ema_20_1d) else ema_20_1d[-1]
        if np.isnan(donch_high_prev) or np.isnan(donch_low_prev) or np.isnan(ema_20_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        donch_high_arr = np.full(len(df_1d), donch_high_prev)
        donch_low_arr = np.full(len(df_1d), donch_low_prev)
        ema_20_arr = np.full(len(df_1d), ema_20_prev)
        donch_high_12h = align_htf_to_ltf(prices, df_1d, donch_high_arr)[i]
        donch_low_12h = align_htf_to_ltf(prices, df_1d, donch_low_arr)[i]
        ema_20_12h = align_htf_to_ltf(prices, df_1d, ema_20_arr)[i]
        
        # 1-week index (14 bars per week: 7*24/12 = 14)
        idx_1w = i // 14
        if idx_1w < 20:  # need enough for ATR MA
            continue
        
        # Get previous 1-week ATR and MA to avoid look-ahead
        atr_prev = atr_1w[idx_1w - 1] if idx_1w - 1 < len(atr_1w) else atr_1w[-1]
        atr_ma_prev = atr_ma_1w[idx_1w - 1] if idx_1w - 1 < len(atr_ma_1w) else atr_ma_1w[-1]
        if np.isnan(atr_prev) or np.isnan(atr_ma_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        atr_arr = np.full(len(df_1w), atr_prev)
        atr_ma_arr = np.full(len(df_1w), atr_ma_prev)
        atr_1w_12h = align_htf_to_ltf(prices, df_1w, atr_arr)[i]
        atr_ma_1w_12h = align_htf_to_ltf(prices, df_1w, atr_ma_arr)[i]
        
        if position == 0:
            # Long: Donchian breakout + low volatility + volume surge
            if (close[i] > donch_high_12h and 
                atr_1w_12h < atr_ma_1w_12h and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: Donchian breakdown + low volatility + volume surge
            elif (close[i] < donch_low_12h and 
                  atr_1w_12h < atr_ma_1w_12h and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price below EMA(20) or volatility expansion
            if close[i] < ema_20_12h or atr_1w_12h > atr_ma_1w_12h * 1.5:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price above EMA(20) or volatility expansion
            if close[i] > ema_20_12h or atr_1w_12h > atr_ma_1w_12h * 1.5:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_Donchian_1wVol_Volume"
timeframe = "12h"
leverage = 1.0