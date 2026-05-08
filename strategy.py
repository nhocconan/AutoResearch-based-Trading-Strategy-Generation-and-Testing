#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate TRIX on daily close (15-period EMA triple smoothed)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = np.where(ema3[:-1] != 0, (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100, 0)
    trix_raw = np.concatenate([np.array([0]), trix_raw])  # align length
    trix_1d = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX to 4h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Chopiness Index on daily high/low (14-period)
    atr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
        atr_1d[i] = tr if i == 1 else (atr_1d[i-1] * 13 + tr) / 14
    
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    
    chop_denom = np.log10((highest_high_14 - lowest_low_14) / sum_atr_14 * np.sqrt(14))
    chop = 100 * chop_denom / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Align Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX > 0 (bullish momentum) + price above weekly EMA50 + chop < 61.8 (trending) + volume spike
            if (trix_1d_aligned[i] > 0 and 
                close[i] > ema_50_1w_aligned[i] and
                chop_aligned[i] < 61.8 and
                vol_ratio[i] > 2.0):
                signals[i] = 0.30
                position = 1
            # Short: TRIX < 0 (bearish momentum) + price below weekly EMA50 + chop < 61.8 (trending) + volume spike
            elif (trix_1d_aligned[i] < 0 and 
                  close[i] < ema_50_1w_aligned[i] and
                  chop_aligned[i] < 61.8 and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: TRIX turns negative OR chop > 61.8 (ranging) OR price below weekly EMA50
            if (trix_1d_aligned[i] < 0 or 
                chop_aligned[i] > 61.8 or
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: TRIX turns positive OR chop > 61.8 (ranging) OR price above weekly EMA50
            if (trix_1d_aligned[i] > 0 or 
                chop_aligned[i] > 61.8 or
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals