#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Aggressive_Confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === Weekly Trend Filter: EMA21 > EMA50 ===
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_up = ema21_1w > ema50_1w
    weekly_down = ema21_1w < ema50_1w
    weekly_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_up)
    weekly_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_down)
    
    # === Daily ATR(14) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 6h Price Action: Price > 6h EMA20 for long, < for short ===
    close_series = pd.Series(prices['close'].values)
    ema20_6h = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === 6h Volume Spike: > 2.0x 20-period average ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema20_val = ema20_6h[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr_1d_aligned[i]
        weekly_up_val = weekly_up_aligned[i]
        weekly_down_val = weekly_down_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema20_val) or np.isnan(vol_ratio_val) or 
            np.isnan(atr_val) or np.isnan(weekly_up_val) or 
            np.isnan(weekly_down_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend + price above EMA20 + volume spike
            if weekly_up_val and close_val > ema20_val and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + price below EMA20 + volume spike
            elif weekly_down_val and close_val < ema20_val and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Weekly trend turns down OR price breaks below EMA20
            if not weekly_up_val or close_val < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Weekly trend turns up OR price breaks above EMA20
            if not weekly_down_val or close_val > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals