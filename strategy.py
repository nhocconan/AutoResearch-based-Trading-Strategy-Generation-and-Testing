#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d Pivot Points (Daily)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = pivot_1d + (range_1d * 1.1 / 12)
    S1 = pivot_1d - (range_1d * 1.1 / 12)
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to daily timeframe (primary)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily data for entry timing
    high_1d_h = df_1d['high'].values
    low_1d_h = df_1d['low'].values
    close_1d_h = df_1d['close'].values
    volume_1d_h = df_1d['volume'].values
    
    # Volume spike detection (20-period daily)
    vol_ma_20 = pd.Series(volume_1d_h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # ATR for volatility filter (14-period daily)
    high_low = high_1d_h - low_1d_h
    high_close = np.abs(high_1d_h - np.roll(close_1d_h, 1))
    low_close = np.abs(low_1d_h - np.roll(close_1d_h, 1))
    high_low[0] = high_1d_h[0] - low_1d_h[0]
    high_close[0] = np.abs(high_1d_h[0] - close_1d_h[0])
    low_close[0] = np.abs(low_1d_h[0] - close_1d_h[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d_h[i]
        vol = volume_1d_h[i]
        trend_up = price > ema34_1w_aligned[i]
        trend_down = price < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: price touches or crosses above S1 with volume confirmation and weekly uptrend
            if (price >= S1_aligned[i] and 
                vol > 2.0 * vol_ma_20_aligned[i] and
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses below R1 with volume confirmation and weekly downtrend
            elif (price <= R1_aligned[i] and 
                  vol > 2.0 * vol_ma_20_aligned[i] and
                  trend_down):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot or ATR-based stop
            if (price < pivot_1d_aligned[i] or 
                price < low_1d_h[i] - 1.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot or ATR-based stop
            if (price > pivot_1d_aligned[i] or 
                price > high_1d_h[i] + 1.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_S1R1_VolumeSpike_1wEMA34Trend"
timeframe = "1d"
leverage = 1.0