#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power > 0 and rising, Bear Power < 0 and falling, price > 12h EMA34 (uptrend), volume spike.
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, price < 12h EMA34 (downtrend), volume spike.
# Exit when power diverges from price or trend changes. Designed to work in both bull and bear markets by measuring trend strength via power imbalance.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.0, ±0.25) to minimize fee churn.

name = "6h_ElderRay_BullBearPower_12hEMA34_6hVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # EMA13 for Elder Ray power calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    # Volume spike: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(34) - trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 and rising (prev bull power < current), Bear Power < 0 and falling (prev bear power > current),
            # price > 12h EMA34 (uptrend), volume spike
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and
                bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and
                close[i] > ema_34_12h_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 and falling (prev bear power < current), Bull Power > 0 and rising (prev bull power > current),
            # price < 12h EMA34 (downtrend), volume spike
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and
                  bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and
                  close[i] < ema_34_12h_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative OR Bear Power turns positive OR price < 12h EMA34 (trend change)
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive OR Bull Power turns negative OR price > 12h EMA34 (trend change)
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals