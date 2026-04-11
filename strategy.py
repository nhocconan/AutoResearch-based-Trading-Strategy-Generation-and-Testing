#!/usr/bin/env python3
# 12h_1d_trix_volume_reversal_v1
# Strategy: 12h TRIX momentum with 1d volume spike and 1d/1w trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: TRIX (TRIple Exponential Average) detects momentum shifts. 
# Volume spike confirms institutional participation. 1d/1w trend filter avoids counter-trend trades.
# Designed for low trade frequency (<30/year) to minimize fee drag. Works in bull/bear via
# momentum reversals aligned with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_trix_volume_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h TRIX (15,9,9) - smoother momentum oscillator
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1w EMA50 for higher timeframe trend
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d volume spike detection (current volume > 2x 20-period average)
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    vol_spike = vol_1d_aligned > 2.0 * vol_avg_20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(trix[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX momentum signals: zero-line cross
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Trend filter: price above/below EMA50 on both 1d and 1w
        price = close[i]
        uptrend_1d = price > ema50_1d_aligned[i]
        uptrend_1w = price > ema50_1w_aligned[i]
        downtrend_1d = price < ema50_1d_aligned[i]
        downtrend_1w = price < ema50_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        # Long: TRIX crosses up AND uptrend on both timeframes AND volume spike
        if trix_cross_up and uptrend_1d and uptrend_1w and vol_spike[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: TRIX crosses down AND downtrend on both timeframes AND volume spike
        elif trix_cross_down and downtrend_1d and downtrend_1w and vol_spike[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite TRIX cross (momentum shift)
        elif position == 1 and trix_cross_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and trix_cross_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals