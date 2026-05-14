#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
# Uses Camarilla pivot levels (R3/S3) from prior 1d for structure, volume spike for conviction,
# and choppiness index (CHOP) to avoid ranging markets. Discrete position sizing (0.0, ±0.30)
# minimizes fee churn. Designed to capture strong breakouts in trending markets while avoiding
# whipsaws in chop. Targets 20-50 trades/year per symbol.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_CHOPFilter_v1"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # Choppiness Index (CHOP) - range: 0-100, >61.8 = range, <38.2 = trend
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (HHV - LLV))) / log10(n)
    # Simplified: use ATR and range over 14 periods
    atr_14 = pd.Series(np.abs(high - low)).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hhvl_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    llvl_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = hhvl_14 - llvl_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * (np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14))
    chop = np.nan_to_num(chop, nan=50.0)  # fill NaN with neutral
    
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (R3, S3) from prior 1d bar
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * camarilla_range / 2.0
    s3_1d = close_1d - 1.1 * camarilla_range / 2.0
    
    # Align to 4h (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(chop[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        if chop[i] >= 61.8:
            # In choppy regime, stay flat or mean revert to mid? We stay flat to avoid whipsaw
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                # Exit long if price touches mid-point (neutral)
                mid_1d = (r3_1d_aligned[i] + s3_1d_aligned[i]) / 2.0
                if abs(close[i] - mid_1d) < 0.001 * close[i]:  # near mid
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                mid_1d = (r3_1d_aligned[i] + s3_1d_aligned[i]) / 2.0
                if abs(close[i] - mid_1d) < 0.001 * close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
            continue
        
        # Trending regime: look for breakouts
        if position == 0:
            # LONG: Price breaks above R3 AND volume spike
            if close[i] > r3_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below S3 AND volume spike
            elif close[i] < s3_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (mean reversion) OR loses volume momentum
            if close[i] < s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (mean reversion) OR loses volume momentum
            if close[i] > r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals