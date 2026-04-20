#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Trix_DoubleSmooth_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # === Daily TRIX (Triple Exponential Average) ===
    close_1d = df_1d['close'].values
    # EMA1
    ema1 = pd.Series(close_1d).ewm(span=12, min_periods=12, adjust=False).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=12, min_periods=12, adjust=False).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=12, min_periods=12, adjust=False).mean().values
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.diff(ema3, prepend=ema3[0]) / np.where(ema3[:-1] != 0, ema3[:-1], np.nan) * 100
    trix_raw = np.append(np.nan, trix_raw[1:])  # align length
    # Smooth TRIX with 12-period EMA
    trix = pd.Series(trix_raw).ewm(span=12, min_periods=12, adjust=False).mean().values
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # === 12h EMA Trend Filter ===
    close_series = pd.Series(prices['close'].values)
    ema34 = close_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema89 = close_series.ewm(span=89, min_periods=89, adjust=False).mean().values
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        trix_val = trix_aligned[i]
        ema34_val = ema34[i]
        ema89_val = ema89[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(trix_val) or np.isnan(ema34_val) or 
            np.isnan(ema89_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume and uptrend
            if trix_val > 0 and vol_ratio_val > 1.8 and ema34_val > ema89_val:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume and downtrend
            elif trix_val < 0 and vol_ratio_val > 1.8 and ema34_val < ema89_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero OR trend breaks down
            if trix_val < 0 or ema34_val < ema89_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero OR trend breaks up
            if trix_val > 0 or ema34_val > ema89_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals