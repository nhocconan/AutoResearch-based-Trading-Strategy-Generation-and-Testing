#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme + 1d ADX regime filter + volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (<-90 or >-10) signal potential reversals.
# In strong trends (ADX > 25 on 1d), these extremes often precede continuations rather than reversals.
# Volume spike (>2.0 x 20-bar average) confirms conviction. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Designed to capture momentum bursts in trending 6h markets while avoiding false signals in ranging conditions.
# Targets 12-30 trades/year per symbol.

name = "6h_WilliamsR_Extreme_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # Williams %R (14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Extreme conditions: oversold (< -90) or overbought (> -10)
    williams_r_oversold = williams_r < -90
    williams_r_overbought = williams_r > -10
    
    # Volume spike: > 2.0 x 20-bar average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX (14) for trend strength on 1d
    high_shift = np.roll(high_1d, 1)
    low_shift = np.roll(low_1d, 1)
    close_shift = np.roll(close_1d, 1)
    high_shift[0] = high_1d[0]
    low_shift[0] = low_1d[0]
    close_shift[0] = close_1d[0]
    
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_shift), np.abs(low_1d - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm = np.where((high_1d - high_shift) > (low_shift - low_1d), np.maximum(high_1d - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low_1d) > (high_1d - high_shift), np.maximum(low_shift - low_1d, 0), 0)
    
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            # In ranging/weak trend, stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                # Exit long if Williams %R returns from extreme (reversion signal)
                if williams_r[i] > -80 and williams_r[i] < -20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if Williams %R returns from extreme
                if williams_r[i] > -80 and williams_r[i] < -20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            continue
        
        # Trending regime: look for extreme Williams %R with volume confirmation
        if position == 0:
            # LONG: Williams %R extremely oversold AND volume spike
            if williams_r_oversold[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R extremely overbought AND volume spike
            elif williams_r_overbought[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns from extreme (loss of momentum)
            if williams_r[i] > -80 and williams_r[i] < -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns from extreme
            if williams_r[i] > -80 and williams_r[i] < -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals