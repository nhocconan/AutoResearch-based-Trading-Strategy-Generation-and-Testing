#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivotBreakout_DailyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend: EMA34 on daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly pivot levels (from weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    H_prev_w = np.roll(high_1w, 1)
    L_prev_w = np.roll(low_1w, 1)
    C_prev_w = np.roll(close_1w, 1)
    H_prev_w[0] = np.nan
    L_prev_w[0] = np.nan
    C_prev_w[0] = np.nan
    
    pivot_w = (H_prev_w + L_prev_w + C_prev_w) / 3
    range_w = H_prev_w - L_prev_w
    
    # Weekly resistance/support levels (using standard pivot multipliers)
    R2_w = pivot_w + range_w
    S2_w = pivot_w - range_w
    
    # Align weekly levels to 6h timeframe
    R2_w_aligned = align_htf_to_ltf(prices, df_1w, R2_w)
    S2_w_aligned = align_htf_to_ltf(prices, df_1w, S2_w)
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    
    # Volume filter: volume > 1.5x 24-period average (6h * 24 = 6 days)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R2_w_aligned[i]) or np.isnan(S2_w_aligned[i]) or 
            np.isnan(pivot_w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R2 + daily uptrend (price > daily EMA34) + volume filter
            long_cond = (close[i] > R2_w_aligned[i]) and \
                        (close[i] > ema_34_1d_aligned[i]) and \
                        volume_filter[i]
            # Short: break below weekly S2 + daily downtrend (price < daily EMA34) + volume filter
            short_cond = (close[i] < S2_w_aligned[i]) and \
                         (close[i] < ema_34_1d_aligned[i]) and \
                         volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below weekly pivot (mean reversion to weekly mean)
            if close[i] < pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above weekly pivot (mean reversion to weekly mean)
            if close[i] > pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals