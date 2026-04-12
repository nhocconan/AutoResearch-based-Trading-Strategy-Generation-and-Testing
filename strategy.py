#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_trix_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15-period triple EMA of ROC)
    roc = df_1d['close'].pct_change(1).values
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100  # Scale for readability
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume confirmation: current 12h volume > 20-period average of daily volume
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_filter = volume > vol_ma_aligned
    
    # Chop index for regime filter (daily)
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(tr_sum / (atr_1d * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop_raw, 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX signal: zero line cross
        trix_prev = trix_aligned[i-1] if i > 0 else 0
        trix_cross_up = trix_prev <= 0 and trix_aligned[i] > 0
        trix_cross_down = trix_prev >= 0 and trix_aligned[i] < 0
        
        # Chop regime: Chop < 50 = trending (favor TRIX signals), Chop > 50 = ranging (avoid)
        trending_regime = chop_aligned[i] < 50
        
        # Long: TRIX crosses above zero in trending market with volume
        long_signal = trix_cross_up and trending_regime and volume_filter[i]
        
        # Short: TRIX crosses below zero in trending market with volume
        short_signal = trix_cross_down and trending_regime and volume_filter[i]
        
        # Exit: chop increases (range) or TRIX returns to zero
        exit_long = (position == 1 and (chop_aligned[i] > 60 or 
                   (trix_aligned[i-1] > 0 and trix_aligned[i] <= 0)))
        exit_short = (position == -1 and (chop_aligned[i] > 60 or 
                     (trix_aligned[i-1] < 0 and trix_aligned[i] >= 0)))
        
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