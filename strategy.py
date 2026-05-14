#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
# Uses Camarilla pivot levels (R1/S1) from prior 1h for structure, EMA50 on 4h for trend direction,
# and volume > 1.5x 20-bar average on 1d for conviction. Discrete position sizing (0.0, ±0.20) to minimize fee churn.
# Designed to capture breakouts in trending markets with institutional volume, avoiding false signals in ranging conditions.
# Targets 15-35 trades/year per symbol by using 4h/1d for signal direction and 1h only for entry timing.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
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
    
    # Precompute session filter (08-20 UTC) once before loop
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 1h Indicators (LTF) ---
    # ATR(14) for stoploss/reference
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # EMA50 on 4h for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Volume spike: >1.5x 20-bar average volume on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # --- 1h Camarilla Levels (from prior 1h bar) ---
    high_shift_1h = np.roll(high, 1)
    low_shift_1h = np.roll(low, 1)
    close_shift_1h = np.roll(close, 1)
    high_shift_1h[0] = high[0]
    low_shift_1h[0] = low[0]
    close_shift_1h[0] = close[0]
    
    camarilla_range = high_shift_1h - low_shift_1h
    r1_1h = close_shift_1h + 1.1 * camarilla_range / 4.0
    s1_1h = close_shift_1h - 1.1 * camarilla_range / 4.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(r1_1h[i]) or
            np.isnan(s1_1h[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when price is above/below 4h EMA50
        if position == 0:
            # LONG: Price breaks above R1 AND price > 4h EMA50 AND 1d volume spike
            if close[i] > r1_1h[i] and close[i] > ema_50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 AND price < 4h EMA50 AND 1d volume spike
            elif close[i] < s1_1h[i] and close[i] < ema_50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (mean reversion to lower level)
            if close[i] < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (mean reversion to upper level)
            if close[i] > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals