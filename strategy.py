#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R4/S4 breakout with 1d volume spike (volume > 2.0x 20-bar SMA) and ADX > 20 trend filter.
# Uses tighter Camarilla levels (R4/S4) for stronger breakout signals, volume confirmation for conviction,
# and ADX > 20 to avoid whipsaws in weak trends. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Designed to capture strong breakouts in trending markets while avoiding false signals.
# Targets 15-30 trades/year per symbol (60-120 total over 4 years) to avoid fee drag.

name = "12h_Camarilla_R4S4_Breakout_1dVolumeSpike_ADXFilter_v1"
timeframe = "12h"
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
    
    # --- 12h Indicators (LTF) ---
    # ADX (14) for trend strength
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm = np.where((high - high_shift) > (low_shift - low), np.maximum(high - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low) > (high - high_shift), np.maximum(low_shift - low, 0), 0)
    
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla levels (R4, S4) from prior 1d bar - tighter levels for stronger signals
    camarilla_range = high_1d - low_1d
    r4_1d = close_1d + 1.1 * camarilla_range
    s4_1d = close_1d - 1.1 * camarilla_range
    
    # Volume spike: current 1d volume > 2.0x 20-bar SMA of volume
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20)
    
    # Align to 12h (wait for completed 1d bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(adx[i]) or
            np.isnan(r4_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 20 (avoid whipsaws in weak trends)
        if adx[i] <= 20:
            # In weak trend/ranging, stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                # Exit long if price touches S4 (mean reversion)
                if close[i] <= s4_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price touches R4 (mean reversion)
                if close[i] >= r4_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            continue
        
        # Strong trend: look for breakouts with volume confirmation
        if position == 0:
            # LONG: Price breaks above R4 AND volume spike
            if close[i] > r4_1d_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 AND volume spike
            elif close[i] < s4_1d_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S4 (mean reversion to lower level)
            if close[i] < s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R4 (mean reversion to upper level)
            if close[i] > r4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals