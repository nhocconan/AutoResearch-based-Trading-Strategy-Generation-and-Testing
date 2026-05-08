#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyPivot_Volume_Squeeze"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels and Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2 * pivot_1w) - low_1w
    s1_1w = (2 * pivot_1w) - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 12h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly Bollinger Bands (20, 2) for squeeze detection
    close_1w_series = pd.Series(close_1w)
    bb_middle = close_1w_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1w_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Align Bollinger width to 12h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly pivot + above daily EMA34 + low volatility (squeeze) + volume confirmation
            if (close[i] > pivot_1w_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                bb_width_aligned[i] < 0.02 and  # Bollinger squeeze
                vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond R2
                if close[i] <= r2_1w_aligned[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: price below weekly pivot + below daily EMA34 + low volatility + volume confirmation
            elif (close[i] < pivot_1w_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  bb_width_aligned[i] < 0.02 and  # Bollinger squeeze
                  vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond S2
                if close[i] >= s2_1w_aligned[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price below weekly pivot OR below daily EMA34 OR high volatility
            if (close[i] < pivot_1w_aligned[i] or 
                close[i] < ema_34_1d_aligned[i] or
                bb_width_aligned[i] > 0.05):  # Exit squeeze
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above weekly pivot OR above daily EMA34 OR high volatility
            if (close[i] > pivot_1w_aligned[i] or 
                close[i] > ema_34_1d_aligned[i] or
                bb_width_aligned[i] > 0.05):  # Exit squeeze
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals