#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme reversal with 1d ADX regime filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) signal potential reversals.
# In strong trends (ADX > 25 on 1d), we fade extremes expecting continuation; in weak trends (ADX <= 25), we fade for mean reversion.
# Volume spike (>1.5x 20-bar average) confirms conviction. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Designed to work in both bull (trend continuation) and bear (mean reversion in ranges) markets. Targets 12-30 trades/year.

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v1"
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
    
    # --- 6h Indicators ---
    # Williams %R (14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Volume spike: >1.5x 20-bar average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX (14) for trend strength
    high_shift_1d = np.roll(high_1d, 1)
    low_shift_1d = np.roll(low_1d, 1)
    close_shift_1d = np.roll(close_1d, 1)
    high_shift_1d[0] = high_1d[0]
    low_shift_1d[0] = low_1d[0]
    close_shift_1d[0] = close_1d[0]
    
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_shift_1d), np.abs(low_1d - close_shift_1d)))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm_1d = np.where((high_1d - high_shift_1d) > (low_shift_1d - low_1d), np.maximum(high_1d - high_shift_1d, 0), 0)
    minus_dm_1d = np.where((low_shift_1d - low_1d) > (high_1d - high_shift_1d), np.maximum(low_shift_1d - low_1d, 0), 0)
    
    plus_di_14_1d = 100 * pd.Series(plus_dm_1d).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_14_1d = 100 * pd.Series(minus_dm_1d).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_14_1d - minus_di_14_1d) / (plus_di_14_1d + minus_di_14_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1d ADX
        if adx_1d_aligned[i] > 25:
            # Strong trend: fade extremes for continuation
            if position == 0:
                # LONG: Extreme oversold AND volume spike
                if williams_r[i] < -80 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Extreme overbought AND volume spike
                elif williams_r[i] > -20 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Extreme overbought (continuation exhausted)
                if williams_r[i] > -20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Extreme oversold (continuation exhausted)
                if williams_r[i] < -80:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Weak trend/ranging: mean reversion from extremes
            if position == 0:
                # LONG: Extreme oversold AND volume spike
                if williams_r[i] < -80 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Extreme overbought AND volume spike
                elif williams_r[i] > -20 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Price returns to neutral (mean reversion complete)
                if williams_r[i] > -50:  # Return to midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Price returns to neutral (mean reversion complete)
                if williams_r[i] < -50:  # Return to midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals