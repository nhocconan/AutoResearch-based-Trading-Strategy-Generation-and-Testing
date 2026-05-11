#!/usr/bin/env python3
# 6h_12h1d_Retracement_Zscore
# Hypothesis: Identifies mean-reversion opportunities during strong trends by measuring
# deviations from the 12h EMA trend using Z-score. In strong trends (ADX>25), prices
# often overextend and revert to the mean. Long when price is significantly below
# the 12h EMA (oversold in uptrend), short when significantly above (overbought in
# downtrend). Uses 1d ADX for trend strength and 60-period Z-score for deviation.
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).

name = "6h_12h1d_Retracement_Zscore"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for EMA trend and 1d data for ADX
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h EMA50 for trend ---
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- 1d ADX for trend strength ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 60-period Z-score of price deviation from 12h EMA ---
    deviation = close - ema_50_12h_aligned
    dev_mean = pd.Series(deviation).rolling(window=60, min_periods=60).mean().values
    dev_std = pd.Series(deviation).rolling(window=60, min_periods=60).std().values
    z_score = (deviation - dev_mean) / (dev_std + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for ADX (30) and Z-score (60)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(z_score[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter: only trade in strong trends
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            if strong_trend:
                # Long: strong uptrend + price significantly below EMA (oversold)
                if ema_50_12h_aligned[i] > close[i] and z_score[i] < -1.5:
                    signals[i] = 0.25
                    position = 1
                # Short: strong downtrend + price significantly above EMA (overbought)
                elif ema_50_12h_aligned[i] < close[i] and z_score[i] > 1.5:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price crosses above EMA OR Z-score reverts to mean
                if close[i] > ema_50_12h_aligned[i] or z_score[i] > -0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses below EMA OR Z-score reverts to mean
                if close[i] < ema_50_12h_aligned[i] or z_score[i] < 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals