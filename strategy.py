#!/usr/bin/env python3
# 1D_TRIX_VolumeSpike_Trend
# Hypothesis: Use TRIX crossover for trend direction on 1d, with volume spike confirmation to enter on pullbacks.
# Long when: TRIX crosses above zero, volume > 2x average, and price pulls back to EMA20.
# Short when: TRIX crosses below zero, volume > 2x average, and price rallies to EMA20.
# Works in bull/bear by following the 1d trend and using volume to confirm institutional interest.
# Target: 15-25 trades/year per symbol.

name = "1D_TRIX_VolumeSpike_Trend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX calculation (15-period EMA of EMA of EMA of close, then ROC)
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    # Rate of change of triple EMA
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # EMA20 for dynamic support/resistance
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(ema20[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        # TRIX crossover signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        if position == 0:
            # Enter long: weekly uptrend + TRIX cross up + volume + price near EMA20
            if weekly_up and trix_cross_up and volume_confirm:
                if close[i] <= ema20[i] * 1.01 and close[i] >= ema20[i] * 0.99:
                    signals[i] = 0.25
                    position = 1
            # Enter short: weekly downtrend + TRIX cross down + volume + price near EMA20
            elif weekly_down and trix_cross_down and volume_confirm:
                if close[i] >= ema20[i] * 0.99 and close[i] <= ema20[i] * 1.01:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions: TRIX turns negative or price moves away from EMA20
            if trix[i] < 0 or close[i] > ema20[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: TRIX turns positive or price moves away from EMA20
            if trix[i] > 0 or close[i] < ema20[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals