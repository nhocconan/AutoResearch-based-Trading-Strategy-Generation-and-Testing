#!/usr/bin/env python3
# 12h_ChaikinMoneyFlow_WeeklyTrend_DailyVolume
# Hypothesis: On 12h timeframe, enter long when Chaikin Money Flow crosses above +0.1 with weekly EMA50 uptrend and daily volume spike (>1.5x 20-period MA).
# Enter short when CMF crosses below -0.1 with weekly EMA50 downtrend and daily volume spike.
# Exit when CMF crosses back towards zero (0.05 for longs, -0.05 for shorts).
# Uses weekly trend filter to avoid counter-trend trades and daily volume to confirm institutional interest.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag and work in both bull and bear markets.

name = "12h_ChaikinMoneyFlow_WeeklyTrend_DailyVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Chaikin Money Flow (20-period)
    # CMF = sum((close - low - (high - close)) / (high - low) * volume) / sum(volume)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)  # avoid division by zero
    mfv = mfm * volume
    cmf = pd.Series(mfv).rolling(window=20, min_periods=20).sum() / \
          pd.Series(volume).rolling(window=20, min_periods=20).sum()
    cmf = cmf.values
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Daily volume confirmation: 20-period moving average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_volume = df_1d['volume'].values
    daily_vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_vol_ma_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(cmf[i]) or np.isnan(weekly_ema50_aligned[i]) or 
            np.isnan(daily_vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        cmf_val = cmf[i]
        weekly_trend = weekly_ema50_aligned[i]
        daily_vol_ma_val = daily_vol_ma_aligned[i]
        daily_vol = df_1d['volume'].values[i // 12] if i >= 12 else 0  # daily volume for current day
        
        # Volume spike: daily volume > 1.5x 20-day average
        vol_spike = daily_vol > daily_vol_ma_val * 1.5
        
        if position == 0:
            # LONG: CMF crosses above +0.1 with weekly uptrend and volume spike
            if cmf_val > 0.1 and weekly_trend > weekly_ema50_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: CMF crosses below -0.1 with weekly downtrend and volume spike
            elif cmf_val < -0.1 and weekly_trend < weekly_ema50_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF crosses back below +0.05 (loss of buying pressure)
            if cmf_val < 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF crosses back above -0.05 (loss of selling pressure)
            if cmf_val > -0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals