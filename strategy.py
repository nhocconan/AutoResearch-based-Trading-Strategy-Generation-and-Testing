#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with volume confirmation and chop regime filter.
# Long when KAMA turns up, price > KAMA, volume > 1.5x avg, and chop > 61.8 (ranging).
# Short when KAMA turns down, price < KAMA, volume > 1.5x avg, and chop > 61.8.
# Exit when price crosses KAMA in opposite direction.
# Uses 1w EMA50 as higher timeframe trend filter (only trade in direction of 1w trend).
# Designed for low-frequency, high-conviction trades in ranging markets (chop > 61.8).
# Discrete position sizing at ±0.25 to minimize fee drag.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_KAMA_VolumeChop_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d KAMA (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility as rolling sum of absolute changes
    volatility = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Choppiness Index (CHOP) on 1d - range: 0-100, >61.8 = ranging
    # True range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    sum_tr14 = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (atr_14 * 14)) / np.log10(14)
    chop = np.where(atr_14 > 0, chop, 50)  # default to 50 when ATR=0
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for KAMA and indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama_aligned[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_chop = chop[i]
        
        # Only trade in choppy markets (chop > 61.8) and in direction of 1w trend
        if curr_chop > 61.8:
            if position == 0:  # Flat - look for new entries
                # Long: KAMA turning up, price > KAMA, volume spike, above 1w EMA
                if (i > start_idx and 
                    kama_aligned[i] > kama_aligned[i-1] and  # KAMA turning up
                    curr_close > curr_kama and 
                    curr_volume_confirm and
                    curr_close > curr_ema_50_1w):
                    signals[i] = 0.25
                    position = 1
                # Short: KAMA turning down, price < KAMA, volume spike, below 1w EMA
                elif (i > start_idx and 
                      kama_aligned[i] < kama_aligned[i-1] and  # KAMA turning down
                      curr_close < curr_kama and 
                      curr_volume_confirm and
                      curr_close < curr_ema_50_1w):
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:  # Long position
                # Exit: price crosses below KAMA
                if curr_close < curr_kama:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:  # Short position
                # Exit: price crosses above KAMA
                if curr_close > curr_kama:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending markets, stay flat
            signals[i] = 0.0
            position = 0
    
    return signals