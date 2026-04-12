#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_trix_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for TRIX and trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # TRIX: Triple Exponential Average (12-period)
    # EMA1 = EMA(close, 12)
    # EMA2 = EMA(EMA1, 12)
    # EMA3 = EMA(EMA2, 12)
    # TRIX = (EMA3 - prev_EMA3) / prev_EMA3 * 100
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3_prev = np.roll(ema3, 1)
    ema3_prev[0] = np.nan
    trix = (ema3 - ema3_prev) / ema3_prev * 100
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align TRIX and EMA50 to 6h
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Chop index for regime filter (6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    chop_raw = 100 * np.log10(tr_sum / (atr_6h * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop_raw, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current 6h volume > 20-day average daily volume
        volume_filter = volume[i] > vol_1d_aligned[i]
        
        # Chop regime: Chop < 38 = trending (trend follow), Chop > 61 = ranging (mean revert)
        trending = chop[i] < 38
        ranging = chop[i] > 61
        
        # TRIX signal: zero-line cross with trend filter
        trix_cross_up = trix_aligned[i] > 0 and trix_aligned[i-1] <= 0
        trix_cross_down = trix_aligned[i] < 0 and trix_aligned[i-1] >= 0
        
        # Long: TRIX crosses above zero in uptrend (price > EMA50) with volume
        long_signal = trix_cross_up and close[i] > ema50_aligned[i] and volume_filter and trending
        
        # Short: TRIX crosses below zero in downtrend (price < EMA50) with volume
        short_signal = trix_cross_down and close[i] < ema50_aligned[i] and volume_filter and trending
        
        # Exit: TRIX returns to zero or chop increases (range)
        exit_long = (position == 1 and (trix_aligned[i] < 0 or ranging))
        exit_short = (position == -1 and (trix_aligned[i] > 0 or ranging))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals