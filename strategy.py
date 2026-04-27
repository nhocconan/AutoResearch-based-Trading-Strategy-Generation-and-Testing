#!/usr/bin/env python3
"""
4h_TRIX_9_VolumeSpike_1dTrend_HTF
Hypothesis: TRIX(9) zero-cross with 1d EMA50 trend filter and volume spike confirmation on 4h.
TRIX filters noise and identifies momentum shifts. 1d EMA50 ensures alignment with higher timeframe trend.
Volume spike confirms breakout authenticity. Designed for 20-40 trades/year on 4h to minimize fee drag.
Works in both bull and bear markets by following 1d trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(9): EMA(EMA(EMA(close,9),9),9) - then ROC
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = 100 * (pd.Series(ema3).pct_change().values)  # ROC of triple EMA
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for TRIX (9*3=27), EMA50, volume average
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or  # need prev for zero-cross
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: TRIX zero-cross in direction of 1d trend with volume spike
            # Long: TRIX crosses above zero AND 1d trend is up (close > EMA50) AND volume spike
            # Short: TRIX crosses below zero AND 1d trend is down (close < EMA50) AND volume spike
            trix_cross_up = trix_prev <= 0 and trix_now > 0
            trix_cross_down = trix_prev >= 0 and trix_now < 0
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if trix_cross_up and trend_up and vol_spike:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif trix_cross_down and trend_down and vol_spike:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when TRIX crosses below zero (momentum loss) OR 1d trend turns down
            if trix_now < 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TRIX crosses above zero (momentum loss) OR 1d trend turns up
            if trix_now > 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_9_VolumeSpike_1dTrend_HTF"
timeframe = "4h"
leverage = 1.0