#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA34 trend filter and 1d ATR-volume spike confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Enter long when Bull Power > 0 and rising, EMA34 > EMA50 (uptrend), and volume spike.
# Enter short when Bear Power > 0 and rising, EMA34 < EMA50 (downtrend), and volume spike.
# Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Designed to capture institutional buying/selling pressure in trending markets with volume confirmation.
# Works in bull (buying power) and bear (selling power) regimes. Targets 12-30 trades/year per symbol.

name = "6h_ElderRay_BullBearPower_12hEMATrend_1dATRVolumeSpike_v1"
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
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # Rising Bull/Bear Power (current > previous)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    # --- 12h Indicators (MTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # EMA34 and EMA50 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Uptrend: EMA34 > EMA50, Downtrend: EMA34 < EMA50
    uptrend = ema34_12h_aligned > ema50_12h_aligned
    downtrend = ema34_12h_aligned < ema50_12h_aligned
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(14) for volume spike confirmation
    high_shift = np.roll(high_1d, 1)
    low_shift = np.roll(low_1d, 1)
    close_shift = np.roll(close_1d, 1)
    high_shift[0] = high_1d[0]
    low_shift[0] = low_1d[0]
    close_shift[0] = close_1d[0]
    
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_shift), np.abs(low_1d - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio = volume / (atr_14 + 1e-10)
    vol_atr_ma_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_atr_ratio > (1.5 * vol_atr_ma_20)
    
    # Align volume spike to 6h
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_rising[i]) or np.isnan(bear_power_rising[i]) or
            np.isnan(uptrend[i]) or np.isnan(downtrend[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 and rising, uptrend, volume spike
            if bull_power[i] > 0 and bull_power_rising[i] and uptrend[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 and rising, downtrend, volume spike
            elif bear_power[i] > 0 and bear_power_rising[i] and downtrend[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 or not rising (loss of buying pressure)
            if bull_power[i] <= 0 or not bull_power_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 or not rising (loss of selling pressure)
            if bear_power[i] <= 0 or not bear_power_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals