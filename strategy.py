#!/usr/bin/env python3
# 6h Elder Ray Power + 1d Trend + Volume Spike
# Hypothesis: Elder Ray (Bull/Bear Power) measures trend strength relative to EMA.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Strong uptrend when Bull Power rising and positive; strong downtrend when Bear Power falling and negative.
# Combined with 1d EMA34 trend filter and volume spikes for confirmation.
# Works in both bull and bear markets by following Elder Ray-defined momentum.
# Designed for low trade frequency (~15-30/year) with clear entry/exit rules.

name = "6h_ElderRay_Power_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Elder Ray Power (EMA13-based) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Smooth the power signals (13-period EMA of the power)
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Volume Spike (20-period on 6h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Rising Bull Power (>0) + volume spike + price above daily EMA34
            if (bull_power_smooth[i] > 0 and 
                bull_power_smooth[i] > bull_power_smooth[i-1] and 
                vol_spike[i] and
                close[i] > ema_34_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Falling Bear Power (<0) + volume spike + price below daily EMA34
            elif (bear_power_smooth[i] < 0 and 
                  bear_power_smooth[i] < bear_power_smooth[i-1] and 
                  vol_spike[i] and
                  close[i] < ema_34_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or stops rising
            if bull_power_smooth[i] <= 0 or bull_power_smooth[i] < bull_power_smooth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive or stops falling
            if bear_power_smooth[i] >= 0 or bear_power_smooth[i] > bear_power_smooth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals