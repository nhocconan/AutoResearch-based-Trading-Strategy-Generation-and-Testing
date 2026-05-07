#!/usr/bin/env python3
name = "4h_TRIX_Trend_Filtered_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for TRIX and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # TRIX: triple smoothed EMA on 12h close
    close_12h = df_12h['close'].values
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix_raw.fillna(0).values
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix, additional_delay_bars=1)
    
    # 12h EMA50 trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: 4-period average on 4h (16h window)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(12, 50, 4)  # Wait for TRIX and EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with uptrend and volume spike
            trix_cross_up = trix_aligned[i] > 0 and trix_aligned[i-1] <= 0
            uptrend = close[i] > ema50_aligned[i]
            vol_spike = volume[i] > vol_ma_4[i] * 1.8
            
            if trix_cross_up and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with downtrend and volume spike
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and not uptrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or trend breaks
            if trix_aligned[i] < 0 or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or trend breaks
            if trix_aligned[i] > 0 or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX momentum with trend filter and volume confirmation
# - TRIX (12,12,12) on 12h captures medium-term momentum
# - Zero-line cross signals momentum shift
# - EMA50 on 12h filters for trend direction (only long in uptrend, short in downtrend)
# - Volume spike (1.8x 4-period average) confirms institutional participation
# - Works in bull markets (buy TRIX crosses up in uptrend) and bear markets (sell TRIX crosses down in downtrend)
# - Exit when TRIX reverses or price breaks trend line
# - Position size 0.25 targets 30-60 trades/year, avoiding fee drag
# - Combines momentum, trend, and volume for robust performance across regimes