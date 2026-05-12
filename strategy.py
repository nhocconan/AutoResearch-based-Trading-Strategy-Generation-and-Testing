#!/usr/bin/env python3
# 1h_4H_Momentum_With_DailyTrend_Filter
# Hypothesis: On 1h timeframe, use 4h momentum (close > 4h open) as signal direction,
# filtered by daily trend (close > daily EMA50). Enter long when both align.
# Enter short when 4h momentum negative and price < daily EMA50.
# Uses volume confirmation to avoid false signals.
# Targets 15-30 trades/year to minimize fee drag.
# Works in bull (momentum + trend) and bear (counter-trend rejections via EMA filter).

name = "1h_4H_Momentum_With_DailyTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for momentum calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h momentum: close > open
    momentum_4h = (df_4h['close'] > df_4h['open']).astype(float).values
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h momentum and daily EMA50 to 1h timeframe
    momentum_4h_aligned = align_htf_to_ltf(prices, df_4h, momentum_4h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 24-period moving average (1 day of 1h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup period
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(momentum_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        mom = momentum_4h_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Positive 4h momentum AND price > daily EMA50 AND volume above average
            if mom > 0.5 and close[i] > ema50 and volume[i] > vol_ma_val:
                signals[i] = 0.20
                position = 1
            # SHORT: Negative 4h momentum AND price < daily EMA50 AND volume above average
            elif mom < 0.5 and close[i] < ema50 and volume[i] > vol_ma_val:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Either momentum turns negative OR price crosses below daily EMA50
            if mom < 0.5 or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Either momentum turns positive OR price crosses above daily EMA50
            if mom > 0.5 or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals