#!/usr/bin/env python3
# Hypothesis: 12h Williams %R extreme with 1d ADX regime filter and volume confirmation.
# Williams %R < -80 indicates oversold (long), > -20 indicates overbought (short).
# Only trade when 1d ADX > 25 (trending market) to avoid whipsaws in ranging conditions.
# Volume confirmation requires current volume > 1.5x 20-bar average volume.
# Discrete position sizing (0.0, ±0.25) minimizes fee churn.
# Designed to capture mean reversion in strong trends while avoiding false signals.
# Targets 15-30 trades/year per symbol.

name = "12h_WilliamsR_Extreme_1dADX_VolumeConfirm_v1"
timeframe = "12h"
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
    
    # --- 12h Indicators (LTF) ---
    # Williams %R (14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Volume confirmation: current volume > 1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
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
    
    # Align ADX to 12h (wait for completed 1d bar)
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
        
        # Regime filter: only trade when 1d ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            # In ranging/weak trend, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Trending regime: look for Williams %R extremes with volume confirmation
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND volume spike
            if williams_r[i] < -80 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND volume spike
            elif williams_r[i] > -20 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (exit oversold) OR volume spike fails
            if williams_r[i] > -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (exit overbought) OR volume spike fails
            if williams_r[i] < -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals