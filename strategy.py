#!/usr/bin/env python3
"""
4h_1d_TRIX_VolumeSpike_TrendFilter
Hypothesis: TRIX (12-period) captures momentum shifts; volume spikes confirm breakout strength; 1d EMA200 filters trend direction for higher win rate in both bull and bear markets. Uses 4h timeframe with low trade frequency (<30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for EMA200 filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    # Daily EMA200 for trend filter
    ema200_daily = pd.Series(close_daily).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_daily_aligned = align_htf_to_ltf(prices, df_daily, ema200_daily)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX on 4h: EMA(EMA(EMA(close,12),12),12) then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Avoid division by zero
    ema3_prev = np.roll(ema3, 1)
    ema3_prev[0] = ema3[0]
    trix = 100 * (ema3 - ema3_prev) / ema3_prev
    
    # Volume spike: current volume > 2.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        start = max(0, i-20)
        vol_ma20[i] = np.mean(volume[start:i]) if i > start else volume[i]
    vol_spike = volume > 2.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if np.isnan(trix[i]) or np.isnan(ema200_daily_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        trix_val = trix[i]
        ema200 = ema200_daily_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Long: TRIX positive crossover + volume spike + price above daily EMA200
            if i > 0 and trix[i-1] <= 0 and trix_val > 0 and vol_ok and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative crossover + volume spike + price below daily EMA200
            elif i > 0 and trix[i-1] >= 0 and trix_val < 0 and vol_ok and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns negative or price closes below daily EMA200
            if trix_val < 0 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns positive or price closes above daily EMA200
            if trix_val > 0 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_TRIX_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0