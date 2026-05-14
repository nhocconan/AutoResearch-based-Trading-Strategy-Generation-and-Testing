#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter.
# Uses Camarilla pivot levels (R3/S3) from prior 1d for structure, volume spike for conviction,
# and ADX > 25 to ensure trending markets (avoid ranging). Discrete position sizing (0.0, ±0.30)
# minimizes fee churn. Designed to capture strong breakouts in trending markets while avoiding
# whipsaws in chop. Targets 12-37 trades/year per symbol.

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXFilter_v1"
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
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ADX for trend strength (14-period)
    # ADX = 100 * smoothed mean of |DI+ - DI-| / (DI+ + DI-)
    # Simplified: use typical ATR-based calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional movement
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (np.abs(plus_di_14) + np.abs(minus_di_14) + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx = np.nan_to_num(adx, nan=20.0)  # fill NaN with neutral
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (R3, S3) from prior 1d bar
    camarilla_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * camarilla_range / 2.0
    s3_1d = close_1d - 1.1 * camarilla_range / 2.0
    
    # Align to 12h (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(adx[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when trending (ADX > 25)
        if adx[i] <= 25:
            # In ranging regime, stay flat to avoid whipsaw
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
            # EXIT LONG: Price crosses below S3 (mean reversion)
            if close[i] < s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (mean reversion)
            if close[i] > r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals