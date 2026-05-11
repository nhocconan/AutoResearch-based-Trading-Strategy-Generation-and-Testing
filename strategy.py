#!/usr/bin/env python3
name = "6h_ChaikinOscillator_1dTrend_Volume"
timeframe = "6h"
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
    
    # 1. Load 1d data ONCE for trend and accumulation/distribution
    df_1d = get_htf_data(prices, '1d')
    
    # 2. 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 3. Calculate Accumulation/Distribution Line on 1d
    clv = ((df_1d['close'] - df_1d['low']) - (df_1d['high'] - df_1d['close'])) / (df_1d['high'] - df_1d['low'])
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    ad_line = (clv * df_1d['volume']).cumsum().values
    
    # 4. Chaikin Oscillator: (3-day EMA of AD) - (10-day EMA of AD)
    ad_series = pd.Series(ad_line)
    ema3 = ad_series.ewm(span=3, min_periods=3, adjust=False).mean().values
    ema10 = ad_series.ewm(span=10, min_periods=10, adjust=False).mean().values
    chaikin_osc = ema3 - ema10
    
    # 5. Align 1d indicators to 6h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    chaikin_osc_aligned = align_htf_to_ltf(prices, df_1d, chaikin_osc)
    
    # 6. Volume filter: 20-period EMA for spike detection on 6h
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # 7. Fixed position size to avoid churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(chaikin_osc_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema200 = close[i] > ema200_1d_aligned[i]
        price_below_ema200 = close[i] < ema200_1d_aligned[i]
        chaikin_positive = chaikin_osc_aligned[i] > 0
        chaikin_negative = chaikin_osc_aligned[i] < 0
        
        if position == 0:
            # Long: Chaikin positive + above 1d EMA200 + volume spike
            if chaikin_positive and price_above_ema200 and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Chaikin negative + below 1d EMA200 + volume spike
            elif chaikin_negative and price_below_ema200 and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Chaikin turns negative OR price crosses below EMA200
                if chaikin_osc_aligned[i] < 0 or close[i] < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Chaikin turns positive OR price crosses above EMA200
                if chaikin_osc_aligned[i] > 0 or close[i] > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals