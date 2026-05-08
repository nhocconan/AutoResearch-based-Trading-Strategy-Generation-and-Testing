#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_Volume_MeanReversion"
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
    
    # Get daily data once for KAMA trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily close for KAMA calculation
    close_1d = df_1d['close'].values
    
    # KAMA: Kaufman Adaptive Moving Average
    # ER = Efficiency Ratio = |change| / sum(|changes|)
    # Smooth = ER * (fastest SC - slowest SC) + slowest SC
    # KAMA[i] = KAMA[i-1] + Smooth * (price[i] - KAMA[i-1])
    def calculate_kama(price_series, fast_sc=2, slow_sc=30):
        price_series = np.asarray(price_series)
        n = len(price_series)
        if n < 2:
            return np.full(n, np.nan)
        
        # Calculate change
        change = np.abs(np.diff(price_series, prepend=price_series[0]))
        
        # Calculate efficiency ratio
        er = np.zeros(n)
        for i in range(1, n):
            if i < 10:  # minimum period for ER calculation
                er[i] = 0
            else:
                net_change = np.abs(price_series[i] - price_series[i-10])
                sum_abs_change = np.sum(change[i-9:i+1])
                er[i] = net_change / sum_abs_change if sum_abs_change != 0 else 0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1))**2
        
        # Calculate KAMA
        kama = np.zeros(n)
        kama[0] = price_series[0]
        for i in range(1, n):
            kama[i] = kama[i-1] + sc[i] * (price_series[i] - kama[i-1])
        
        return kama
    
    # Calculate KAMA on daily close
    kama = calculate_kama(close_1d, fast_sc=2, slow_sc=30)
    kama_trend = (close_1d > kama).astype(float)
    kama_trend_aligned = align_htf_to_ltf(prices, df_1d, kama_trend)
    
    # Daily volume spike: current volume > 2.0 * 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma20d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma20d[i] = np.mean(volume_1d[i-20:i])
    vol_spike = volume_1d > (vol_ma20d * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Mean reversion signal: price deviation from KAMA
    # Long when price < KAMA * 0.98 (oversold)
    # Short when price > KAMA * 1.02 (overbought)
    kama_ratio = close_1d / kama
    kama_ratio_aligned = align_htf_to_ltf(prices, df_1d, kama_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_trend_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(kama_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price oversold AND daily uptrend AND volume spike
            long_cond = (kama_ratio_aligned[i] < 0.98 and 
                        kama_trend_aligned[i] > 0.5 and 
                        vol_spike_aligned[i])
            
            # Short entry: price overbought AND daily downtrend AND volume spike
            short_cond = (kama_ratio_aligned[i] > 1.02 and 
                         kama_trend_aligned[i] < 0.5 and 
                         vol_spike_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to KAMA (mean reversion)
            if kama_ratio_aligned[i] >= 1.00:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to KAMA (mean reversion)
            if kama_ratio_aligned[i] <= 1.00:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend with volume confirmation and mean reversion entries.
# KAMA adapts to market conditions - fast in trends, slow in ranges.
# Enters mean reversion trades when price deviates significantly from KAMA.
# Volume spike confirms momentum behind the move.
# Works in both bull (trend following) and bear (mean reversion) markets.
# Target: 20-35 trades/year to minimize fee decay while capturing significant moves.