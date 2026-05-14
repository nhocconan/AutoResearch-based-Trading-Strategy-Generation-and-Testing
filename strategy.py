#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and ATR-based volume spike.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA34 (uptrend) AND volume spike.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA34 (downtrend) AND volume spike.
# Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
# Works in bull via trend continuation, in bear via mean reversion from extreme power readings.

name = "6h_ElderRay_BullBearPower_1dEMA34_ATRVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Smoothed Elder Ray for trend confirmation (2-period smoothing)
    bull_power_smooth = pd.Series(bull_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    # ATR(14) for volatility and volume spike calculation
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio = volume / (atr_14 + 1e-10)
    vol_atr_ma_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_atr_ratio > (1.5 * vol_atr_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # EMA34 on 1d for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # LONG: Bull Power positive AND rising AND uptrend AND volume spike
            if (bull_power[i] > 0 and 
                bull_power_smooth[i] > bull_power_smooth[i-1] and 
                uptrend and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power negative AND falling (more negative) AND downtrend AND volume spike
            elif (bear_power[i] < 0 and 
                  bear_power_smooth[i] < bear_power_smooth[i-1] and 
                  downtrend and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative OR power deteriorates
            if bull_power[i] <= 0 or bull_power_smooth[i] < bull_power_smooth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive OR power deteriorates
            if bear_power[i] >= 0 or bear_power_smooth[i] > bear_power_smooth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals