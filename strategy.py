#!/usr/bin/env python3
name = "4h_Trix_Volume_Chop_Trend_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # TRIX calculation (15-period)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3
    trix_smoothed = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Chopiness Index (14-period) on daily
    atr1 = np.maximum(high_1d - low_1d,
                      np.maximum(np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]])),
                                 np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))))
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_h = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_h - min_l)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after indicators are ready
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if daily trend data is not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for TRIX signal with volume and chop confirmation
            if volume_filter[i] and chop_aligned[i] < 50:  # Trending market (chop < 50)
                if trix_smoothed[i] > 0 and close[i] > ema34_1d_aligned[i]:
                    # TRIX positive + price above daily EMA34 = long
                    signals[i] = 0.25
                    position = 1
                elif trix_smoothed[i] < 0 and close[i] < ema34_1d_aligned[i]:
                    # TRIX negative + price below daily EMA34 = short
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long when TRIX turns negative or chop increases
            if trix_smoothed[i] < 0 or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when TRIX turns positive or chop increases
            if trix_smoothed[i] > 0 or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals