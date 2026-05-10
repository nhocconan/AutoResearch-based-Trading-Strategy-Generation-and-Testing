#!/usr/bin/env python3
# 1h_4h1d_TRIX_Momentum_Filter
# Hypothesis: TRIX (12) crossover on 1h with 4h/1d trend alignment and volume confirmation.
# TRIX filters noise; long when TRIX > 0 and 4h/1d uptrend, short when TRIX < 0 and 4h/1d downtrend.
# Uses 4h/1d for trend direction, 1h for entry timing. Targets 60-150 trades over 4 years via TRIX smoothing + trend filter.
# Works in bull/bear by requiring trend alignment, avoiding counter-trend traps.

name = "1h_4h1d_TRIX_Momentum_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and 1d data for trend and structure
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX (1-period ROC of triple-smoothed EMA)
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    
    # 4h EMA50 and EMA200 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Aligned close prices for trend checks
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h['close'].values)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Volume average (24-period for 1h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for TRIX + EMAs + vol MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_200_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 4h EMA50 > EMA200 AND 1h close > 4h EMA50 AND 1d close > 1d EMA50
        uptrend_4h = close_4h_aligned[i] > ema_50_4h_aligned[i] and ema_50_4h_aligned[i] > ema_200_4h_aligned[i]
        uptrend_1d = close_1d_aligned[i] > ema_50_1d_aligned[i]
        uptrend = uptrend_4h and uptrend_1d
        
        # Downtrend: 4h EMA50 < EMA200 AND 1h close < 4h EMA50 AND 1d close < 1d EMA50
        downtrend_4h = close_4h_aligned[i] < ema_50_4h_aligned[i] and ema_50_4h_aligned[i] < ema_200_4h_aligned[i]
        downtrend_1d = close_1d_aligned[i] < ema_50_1d_aligned[i]
        downtrend = downtrend_4h and downtrend_1d
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: TRIX > 0 in uptrend with volume
            if trix[i] > 0 and uptrend and volume_surge:
                signals[i] = 0.20
                position = 1
            # Short: TRIX < 0 in downtrend with volume
            elif trix[i] < 0 and downtrend and volume_surge:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Long exit: TRIX <= 0 or trend fails
                if trix[i] <= 0 or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short exit: TRIX >= 0 or trend fails
                if trix[i] >= 0 or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals