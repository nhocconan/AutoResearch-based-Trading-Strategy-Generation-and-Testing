#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA trend filter and 1d ATR volume spike.
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low.
# Long when Bull Power > 0, Bear Power < 0, price > 12h EMA50, and volume spike (>1.5x 20-bar ATR-scaled avg volume).
# Short when Bear Power > 0, Bull Power < 0, price < 12h EMA50, and volume spike.
# Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Designed to capture strong directional moves
# with institutional participation (volume spike) in trending markets (12h EMA50 filter). Targets 12-30 trades/year per symbol.

name = "6h_ElderRay_BullBear_12hEMA50_1dATRVolumeSpike_v1"
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
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # --- 12h Indicators (MTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # EMA50 on 12h for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(14) for volume normalization on 1d
    high_shift = np.roll(high_1d, 1)
    low_shift = np.roll(low_1d, 1)
    close_shift = np.roll(close_1d, 1)
    high_shift[0] = high_1d[0]
    low_shift[0] = low_1d[0]
    close_shift[0] = close_1d[0]
    
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_shift), np.abs(low_1d - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio = volume / (atr_14 + 1e-10)  # Use LTF volume with HTF ATR for regime-adjusted volume
    vol_atr_ma_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_atr_ratio > (1.5 * vol_atr_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # LONG: Bull Power > 0, Bear Power < 0, price > 12h EMA50, volume spike
        if (bull_power[i] > 0 and bear_power[i] < 0 and 
            close[i] > ema50_12h_aligned[i] and volume_spike[i]):
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # SHORT: Bear Power > 0, Bull Power < 0, price < 12h EMA50, volume spike
        elif (bear_power[i] > 0 and bull_power[i] < 0 and 
              close[i] < ema50_12h_aligned[i] and volume_spike[i]):
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # EXIT: Power divergence or loss of volume confirmation
        else:
            # Exit long if Bull Power turns negative or Bear Power positive (momentum loss)
            if position == 1 and (bull_power[i] <= 0 or bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            # Exit short if Bear Power turns negative or Bull Power positive (momentum loss)
            elif position == -1 and (bear_power[i] <= 0 or bull_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            # Flat or hold
            else:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals