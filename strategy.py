#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1d volume spike and 1d chop regime filter.
# Long when KAMA direction is up AND 1d volume > 1.8x 20-period average AND 1d chop > 61.8 (range market).
# Short when KAMA direction is down AND 1d volume > 1.8x 20-period average AND 1d chop > 61.8.
# Exit when KAMA direction reverses or price crosses 12h KAMA.
# Uses discrete position size 0.25. Designed to capture mean reversion in ranging markets with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: KAMA (ER=10, SC=2,30) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Efficiency Ratio
    change_12h = np.abs(np.diff(close_12h, n=10))  # 10-period change
    volatility_12h = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)[:len(change_12h)]  # 10-period volatility
    er_12h = np.divide(change_12h, volatility_12h, out=np.zeros_like(change_12h), where=volatility_12h!=0)
    er_12h = np.concatenate([np.full(10, np.nan), er_12h])  # align length
    
    # Smoothing Constants
    sc_12h = 2.0 / (30 + 1)  # fast SC
    sc_slow_12h = 2.0 / (2 + 1)  # slow SC
    sc_12h = er_12h * (sc_12h - sc_slow_12h) + sc_slow_12h
    sc_12h = sc_12h ** 2
    
    # KAMA
    kama_12h = np.full_like(close_12h, np.nan)
    kama_12h[10] = close_12h[10]  # seed
    for i in range(11, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    
    # KAMA direction (1=up, -1=down, 0=flat)
    kama_dir_12h = np.diff(kama_12h, n=1)
    kama_dir_12h = np.concatenate([[0], np.sign(kama_dir_12h)])
    kama_dir_12h = np.where(np.abs(kama_dir_12h) < 1e-10, 0, kama_dir_12h)
    
    # Align 12h KAMA direction to lower timeframe
    kama_dir_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_dir_12h)
    
    # === 1d Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    # === 1d Indicators: Choppiness Index (CHOP > 61.8 = range) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest High and Lowest Low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    sum_tr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_tr_1d / (hh_1d - ll_1d)) / np.log10(14)
    chop_1d = np.where((hh_1d - ll_1d) == 0, 50, chop_1d)  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_range = chop_1d_aligned > 61.8
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(kama_dir_12h_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_range[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        kama_dir = kama_dir_12h_aligned[i]
        vol_spike = volume_spike[i]
        is_chop = chop_range[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if KAMA direction turns down
            if kama_dir <= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if KAMA direction turns up
            if kama_dir >= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: KAMA up AND volume spike AND choppy market (mean reversion setup)
            if kama_dir > 0 and vol_spike and is_chop:
                signals[i] = 0.25
                position = 1
            
            # SHORT: KAMA down AND volume spike AND choppy market (mean reversion setup)
            elif kama_dir < 0 and vol_spike and is_chop:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_KAMA10_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0