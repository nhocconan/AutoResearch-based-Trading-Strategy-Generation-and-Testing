#!/usr/bin/env python3
# 1D_1W_TRIX_ZERO_CROSS_VOLUME
# Hypothesis: Daily TRIX zero-cross with weekly trend filter and volume confirmation.
# TRIX(12) crossing zero indicates momentum shift. Weekly trend filter ensures trades align with higher timeframe direction.
# Volume spike (>2x 20-day average) confirms momentum strength.
# Works in bull markets (buy on bullish TRIX cross in uptrend) and bear markets (sell on bearish TRIX cross in downtrend).
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1D_1W_TRIX_ZERO_CROSS_VOLUME"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily TRIX calculation
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    trix_prev = np.roll(trix, 1)
    trix_prev[0] = 0
    trix_cross_up = (trix > 0) & (trix_prev <= 0)
    trix_cross_down = (trix < 0) & (trix_prev >= 0)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for TRIX and EMA
        if np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses up + volume spike + price above weekly EMA34 (uptrend)
            if trix_cross_up[i] and volume_spike[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses down + volume spike + price below weekly EMA34 (downtrend)
            elif trix_cross_down[i] and volume_spike[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses down OR price crosses below weekly EMA34
            if trix_cross_down[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses up OR price crosses above weekly EMA34
            if trix_cross_up[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals