#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1wTrend_Filter
# Hypothesis: Elder Ray index (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h combined with 1-week EMA trend filter.
# In bull markets (price > weekly EMA), we take long signals when Bull Power turns positive after being negative.
# In bear markets (price < weekly EMA), we take short signals when Bear Power turns negative after being positive.
# This captures momentum shifts in the direction of the higher timeframe trend, working in both bull and bear regimes.
# Uses volume confirmation to avoid false signals. Designed for low trade frequency (~20-50/year) to minimize fee drag.

name = "6h_ElderRay_BullBearPower_1wTrend_Filter"
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
    
    # === 1-week Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 30-period EMA on 1w for trend direction
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # === Elder Ray on 6h (EMA13) ===>
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_30_1w_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1w EMA
        bullish_trend = close[i] > ema_30_1w_aligned[i]
        bearish_trend = close[i] < ema_30_1w_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Bull Power turns positive (from negative or zero) in bullish trend with volume
            if bullish_trend and vol_ok and bull_power[i] > 0 and bull_power[i-1] <= 0:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power turns negative (from positive or zero) in bearish trend with volume
            elif bearish_trend and vol_ok and bear_power[i] < 0 and bear_power[i-1] >= 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or trend turns bearish
            if bull_power[i] < 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive or trend turns bullish
            if bear_power[i] > 0 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals