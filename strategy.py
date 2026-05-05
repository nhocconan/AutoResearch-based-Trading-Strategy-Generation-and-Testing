#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h EMA trend filter + volume spike
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when: Bull Power > 0 AND Bear Power rising (less negative) AND 12h EMA50 > EMA200 AND volume > 2x 20-period MA
# Short when: Bear Power < 0 AND Bull Power falling (less positive) AND 12h EMA50 < EMA200 AND volume > 2x 20-period MA
# Exit when: Power signals reverse OR volume drops below average
# Uses Elder Ray for momentum, 12h EMA for trend filter, volume for conviction
# Timeframe: 6h, HTF: 12h. Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.

name = "6h_ElderRay_12hEMA_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray on 6h
    if len(close) >= 13:
        ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema13 = np.full(n, np.nan)
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Power momentum (change in power)
    bull_power_momentum = np.diff(bull_power, prepend=np.nan)
    bear_power_momentum = np.diff(bear_power, prepend=np.nan)
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 on 12h
    close_12h = df_12h['close'].values
    if len(close_12h) >= 200:
        ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    elif len(close_12h) >= 50:
        ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema200_12h = np.full(len(close_12h), np.nan)
    else:
        ema50_12h = np.full(len(close_12h), np.nan)
        ema200_12h = np.full(len(close_12h), np.nan)
    
    # 12h trend: bullish when EMA50 > EMA200
    ema50_gt_ema200 = np.zeros(len(ema50_12h), dtype=bool)
    ema50_lt_ema200 = np.zeros(len(ema50_12h), dtype=bool)
    for i in range(len(ema50_12h)):
        if not np.isnan(ema50_12h[i]) and not np.isnan(ema200_12h[i]):
            ema50_gt_ema200[i] = ema50_12h[i] > ema200_12h[i]
            ema50_lt_ema200[i] = ema50_12h[i] < ema200_12h[i]
    
    # Align 12h EMA trend to 6h timeframe
    ema50_gt_ema200_aligned = align_htf_to_ltf(prices, df_12h, ema50_gt_ema200.astype(float))
    ema50_lt_ema200_aligned = align_htf_to_ltf(prices, df_12h, ema50_lt_ema200.astype(float))
    
    # Volume confirmation on 6h: volume > 2x 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_momentum[i]) or np.isnan(bear_power_momentum[i]) or
            np.isnan(ema50_gt_ema200_aligned[i]) or np.isnan(ema50_lt_ema200_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bull Power rising AND 12h uptrend AND volume spike
            if (bull_power[i] > 0 and 
                bull_power_momentum[i] > 0 and 
                ema50_gt_ema200_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bear Power falling (more negative) AND 12h downtrend AND volume spike
            elif (bear_power[i] < 0 and 
                  bear_power_momentum[i] < 0 and 
                  ema50_lt_ema200_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bull Power falling OR 12h downtrend
            if (bull_power[i] <= 0 or 
                bull_power_momentum[i] <= 0 or 
                ema50_lt_ema200_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Bear Power rising OR 12h uptrend
            if (bear_power[i] >= 0 or 
                bear_power_momentum[i] >= 0 or 
                ema50_gt_ema200_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals