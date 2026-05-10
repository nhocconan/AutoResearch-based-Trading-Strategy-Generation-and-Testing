#!/usr/bin/env python3
# 6h_ElderRay_Signal_1wTrend_Filter
# Hypothesis: Uses Elder Ray (Bull/Bear Power) on 6h with 1-week trend filter to capture trend-following moves in BTC/ETH.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low. Enter long when Bull Power > 0 and 1w EMA(34) rising.
# Enter short when Bear Power > 0 and 1w EMA(34) falling. Uses volume confirmation and discrete position sizing (0.25) to reduce churn.
# Designed to work in both bull and bear markets by aligning with 1-week trend and avoiding counter-trend trades.

name = "6h_ElderRay_Signal_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_slope = ema_34_1w[1:] - ema_34_1w[:-1]  # daily slope
    ema_34_1w_slope = np.concatenate([[False], ema_34_1w_slope > 0])  # rising if slope > 0
    ema_34_1w_slope = np.concatenate([ema_34_1w_slope, [False]])  # align length
    ema_34_1w_rising = ema_34_1w_slope
    ema_34_1w_falling = ~ema_34_1w_slope
    
    # Align 1w trend to 6h
    ema_34_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_rising)
    ema_34_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_falling)
    
    # Calculate Elder Ray on 6h: EMA(13) of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = ema_13 - low   # Bear Power = EMA - Low
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 13)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_threshold[i]) or \
           np.isnan(ema_34_1w_rising_aligned[i]) or np.isnan(ema_34_1w_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power > 0 + 1w EMA(34) rising + volume spike
            if (bull_power[i] > 0 and 
                ema_34_1w_rising_aligned[i] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 + 1w EMA(34) falling + volume spike
            elif (bear_power[i] > 0 and 
                  ema_34_1w_falling_aligned[i] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or 1w trend turns down or volume drops
            if (bull_power[i] <= 0 or 
                not ema_34_1w_rising_aligned[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 or 1w trend turns up or volume drops
            if (bear_power[i] <= 0 or 
                not ema_34_1w_falling_aligned[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals