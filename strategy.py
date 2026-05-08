#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullBearPower_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA34 trend filter (using close)
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = (close_1w > ema34_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Elder Ray indicators on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike detection: current volume > 2.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for EMA13 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13[i]) or np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power > 0 (bullish momentum) with volume spike and 1w uptrend
            long_cond = (bull_power[i] > 0 and vol_spike[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: Bear Power < 0 (bearish momentum) with volume spike and 1w downtrend
            short_cond = (bear_power[i] < 0 and vol_spike[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power < 0 (momentum shifts bearish)
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power > 0 (momentum shifts bullish)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) strategy with 1-week trend filter and volume spike confirmation on 6h timeframe.
# Bull Power > 0 indicates bullish momentum (high > EMA13), Bear Power < 0 indicates bearish momentum (low < EMA13).
# Volume spike confirms institutional participation. 1w trend filter ensures alignment with higher timeframe trend.
# Works in both bull and bear markets by following the 1w trend direction with momentum confirmation.
# Target: 15-25 trades/year (60-100 total over 4 years) to avoid excessive trading.
# Uses discrete sizing (0.25) to minimize churn from frequent signal changes. Focus on BTC/ETH performance.