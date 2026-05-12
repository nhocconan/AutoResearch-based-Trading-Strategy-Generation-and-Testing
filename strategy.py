#!/usr/bin/env python3
# 6h Elder Ray Power + Daily Trend + Volume Spike
# Hypothesis: Elder Ray (Bull/Bear Power) measures bullish/bearish strength relative to EMA.
# Bull Power = High - EMA, Bear Power = Low - EMA.
# Combines with 1d EMA trend filter and volume spikes for momentum confirmation.
# Works in bull markets via Bull Power > 0 and in bear markets via Bear Power < 0.
# Low trade frequency expected due to strict confluence requirements.

name = "6h_ElderRay_Power_Trend_Volume"
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
    
    # === 1d Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Elder Ray (13-period EMA) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA
    bear_power = low - ema13   # Bear Power = Low - EMA
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (bullish strength) + price above 1d EMA34 + volume spike
            if bull_power[i] > 0 and close[i] > ema_34_6h[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bearish strength) + price below 1d EMA34 + volume spike
            elif bear_power[i] < 0 and close[i] < ema_34_6h[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (bullish strength faded) or trend change
            if bull_power[i] <= 0 or close[i] < ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (bearish strength faded) or trend change
            if bear_power[i] >= 0 or close[i] > ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals