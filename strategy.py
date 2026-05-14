#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d volume spike filter and session timing (08-20 UTC).
# Uses 4h Camarilla levels for structure, 1d ATR-normalized volume spike (>1.5x 20-bar average) for conviction,
# and restricts trading to active session hours (08-20 UTC) to reduce noise. Discrete position sizing (0.0, ±0.20).
# Designed to capture institutional breakouts during liquid hours while avoiding false signals in low-volume periods.
# Targets 15-35 trades/year per symbol to stay within fee drag limits.

name = "1h_Camarilla_R3S3_Breakout_1dVolumeSpike_SessionFilter_v1"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 4h Indicators (MTF for direction) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels (R3, S3) from prior 4h bar
    camarilla_range_4h = high_4h - low_4h
    r3_4h = close_4h + 1.1 * camarilla_range_4h / 2.0
    s3_4h = close_4h - 1.1 * camarilla_range_4h / 2.0
    
    # Align to 1h (wait for completed 4h bar)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # --- 1d Indicators (HTF for filter) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # ATR(14) for 1d volatility normalization
    high_shift_1d = np.roll(high_1d, 1)
    low_shift_1d = np.roll(low_1d, 1)
    close_shift_1d = np.roll(close_1d, 1)
    high_shift_1d[0] = high_1d[0]
    low_shift_1d[0] = low_1d[0]
    close_shift_1d[0] = close_1d[0]
    
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - close_shift_1d), 
                                  np.abs(low_1d - close_shift_1d)))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio_1d = volume_1d / (atr_14_1d + 1e-10)
    vol_atr_ma_20_1d = pd.Series(vol_atr_ratio_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_atr_ratio_1d > (1.5 * vol_atr_ma_20_1d)
    
    # Align volume spike to 1h (wait for completed 1d bar)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session
        if (np.isnan(r3_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3_4h AND 1d volume spike
            if close[i] > r3_4h_aligned[i] and volume_spike_1d_aligned[i] > 0.5:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3_4h AND 1d volume spike
            elif close[i] < s3_4h_aligned[i] and volume_spike_1d_aligned[i] > 0.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3_4h (mean reversion to lower level)
            if close[i] < s3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above R3_4h (mean reversion to upper level)
            if close[i] > r3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals