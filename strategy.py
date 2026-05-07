#!/usr/bin/env python3
name = "12h_TRIX_Volume_Spike_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for TRIX and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 18:
        return np.zeros(n)
    
    # TRIX(12) = EMA(EMA(EMA(close,12),12),12) - 1 period rate of change
    close_series = pd.Series(df_1d['close'])
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # Percentage change
    trix_values = trix.values
    
    # Daily EMA(34) for trend filter
    ema_34_1d = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align TRIX and EMA to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_values)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2-period average (1 day of 12h bars)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume and daily uptrend
            trix_cross_up = trix_aligned[i] > 0 and trix_aligned[i-1] <= 0
            vol_condition = volume[i] > vol_ma_2[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if trix_cross_up and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume and daily downtrend
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or volume drops
            if trix_aligned[i] < 0 or volume[i] < vol_ma_2[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or volume drops
            if trix_aligned[i] > 0 or volume[i] < vol_ma_2[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX(12) zero-cross with volume spike and daily trend filter
# - TRIX shows momentum changes; zero-cross signals trend shifts
# - Volume spike (2x 2-period average) confirms institutional participation
# - Daily EMA(34) trend filter ensures trades align with higher timeframe trend
# - Works in bull (TRIX up-cross in uptrend) and bear (TRIX down-cross in downtrend)
# - Exit on TRIX reverse cross or volume weakening
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Novel application: TRIX momentum with volume confirmation on 12h timeframe
# - Avoids overtrading by requiring multiple confluence factors
# - Uses daily TRIX and trend for stability, 12h for execution timing
# - Volume confirmation reduces false signals in choppy markets