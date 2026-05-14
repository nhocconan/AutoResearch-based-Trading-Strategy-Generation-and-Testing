#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA trend filter and ATR-based volume spike.
# Uses Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) to measure bull/bear strength,
# combined with 1d EMA34 trend filter to ensure alignment with higher timeframe direction.
# ATR-normalized volume spike (>1.8x 20-bar average) adds confirmation for institutional participation.
# Discrete position sizing (0.0, ±0.25) minimizes fee churn. Designed to capture strong momentum
# moves in both bull and bear markets by trading with the 1d trend when Elder Ray shows extreme
# power and volume confirms. Targets 12-30 trades/year per symbol.

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
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # ATR(14) for volatility normalization
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
    volume_spike = vol_atr_ratio > (1.8 * vol_atr_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # EMA34 for trend direction on 1d
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend direction
        is_uptrend = close[i] > ema34_1d_aligned[i]
        is_downtrend = close[i] < ema34_1d_aligned[i]
        
        # Exit logic (applies in all regimes)
        if position == 1:  # Long position
            # Exit long if bear power exceeds bull power (momentum shift) OR close below EMA34_1d
            if bear_power[i] > bull_power[i] or close[i] <= ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if bull power exceeds bear power (momentum shift) OR close above EMA34_1d
            if bull_power[i] > bear_power[i] or close[i] >= ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # LONG: Bull power > bear power (bullish momentum) AND volume spike AND 1d uptrend
            if bull_power[i] > bear_power[i] and volume_spike[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear power > bull power (bearish momentum) AND volume spike AND 1d downtrend
            elif bear_power[i] > bull_power[i] and volume_spike[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals