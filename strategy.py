#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_WeeklyTrend
Hypothesis: Use TRIX (1-period rate-of-change of triple EMA) on weekly timeframe for trend direction, combined with volume spike on 12h for entry timing. Long when weekly TRIX > 0 and 12h price breaks above 12-period high with volume spike; short when weekly TRIX < 0 and price breaks below 12-period low with volume spike. Designed for low trade frequency (~20-40/year) to avoid fee drag, works in both bull (riding weekly uptrend) and bear (selling weekly downtrend) markets.
"""

name = "12h_TRIX_VolumeSpike_WeeklyTrend"
timeframe = "12h"
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
    
    # === Weekly Data for TRIX Trend Filter ===
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate TRIX: 1-period ROC of triple EMA (15-period each)
    ema1 = pd.Series(close_weekly).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    # TRIX = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix_raw = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix_raw.fillna(0).values  # Handle NaN at start
    
    # Align weekly TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_weekly, trix)
    
    # === 12h Indicators for Entry Timing ===
    # 12-period high/low for breakout detection
    high_12 = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_12 = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if np.isnan(trix_aligned[i]) or np.isnan(high_12[i]) or np.isnan(low_12[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly TRIX > 0 (uptrend) AND price breaks above 12-period high AND volume spike
            if trix_aligned[i] > 0 and close[i] > high_12[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly TRIX < 0 (downtrend) AND price breaks below 12-period low AND volume spike
            elif trix_aligned[i] < 0 and close[i] < low_12[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly TRIX turns negative OR price breaks below 12-period low
            if trix_aligned[i] < 0 or close[i] < low_12[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: weekly TRIX turns positive OR price breaks above 12-period high
            if trix_aligned[i] > 0 or close[i] > high_12[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals