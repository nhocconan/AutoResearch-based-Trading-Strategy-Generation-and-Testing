#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w KAMA trend direction + 1d volume spike + chop regime filter
# - Uses 1w HTF for KAMA(10,2,30) to identify primary trend direction (avoid counter-trend trades)
# - Long when 1w KAMA rising AND 1d volume > 2.0x 20-period average AND chop(14) > 61.8 (range regime)
# - Short when 1w KAMA falling AND 1d volume > 2.0x 20-period average AND chop(14) > 61.8 (range regime)
# - Fixed position size 0.25 to control drawdown
# - Chop regime filter ensures we only trade in ranging markets where mean reversion works
# - Volume spike confirms institutional participation at range boundaries
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)

name = "12h_1w_kama_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w KAMA (10,2,30) - Kaufman Adaptive Moving Average
    # ER = |Close - Close[10]| / Sum|Close[i] - Close[i-1]| for i=1 to 10
    # Smooth = (ER * (fast - slow) + slow)^2 where fast=2/(2+1), slow=2/(30+1)
    change_1w = np.abs(close_1w - np.roll(close_1w, 10))
    change_1w[:10] = np.nan  # First 10 values invalid
    
    abs_diff_1w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    sum_abs_diff_1w = pd.Series(abs_diff_1w).rolling(window=10, min_periods=10).sum().values
    
    er_1w = change_1w / sum_abs_diff_1w
    er_1w = np.where(sum_abs_diff_1w == 0, 0, er_1w)  # Avoid division by zero
    
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (30.0 + 1.0)
    sc_1w = (er_1w * (fast_sc - slow_sc) + slow_sc) ** 2
    sc_1w = np.nan_to_num(sc_1w, nan=slow_sc**2)  # Fill NaN with slow SC squared
    
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[9] = close_1w[9]  # Initialize with first valid close
    for i in range(10, len(close_1w)):
        if not np.isnan(sc_1w[i]):
            kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
        else:
            kama_1w[i] = kama_1w[i-1]
    
    # Align KAMA to 12h timeframe (wait for completed 1w bar)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1w KAMA direction (rising/falling)
    kama_dir_1w = np.diff(kama_1w_aligned, prepend=kama_1w_aligned[0])
    kama_rising = kama_dir_1w > 0
    kama_falling = kama_dir_1w < 0
    
    # Load 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Chopiness Index (14-period)
    # Chop = 100 * log10(sum(TR) / (ATR * N)) / log10(N)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    chop_1d = 100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14)
    chop_1d = np.where(atr_1d == 0, 50, chop_1d)  # Default to middle when ATR=0
    chop_1d = np.nan_to_num(chop_1d, nan=50.0)
    
    # Align chop to 12h timeframe (wait for completed 1d bar)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if volume_confirmed and chop_filter:
            # Long entry: 1w KAMA rising (uptrend) + volume spike + chop regime
            if kama_rising[i]:
                signals[i] = 0.25
            # Short entry: 1w KAMA falling (downtrend) + volume spike + chop regime
            elif kama_falling[i]:
                signals[i] = -0.25
    
    return signals