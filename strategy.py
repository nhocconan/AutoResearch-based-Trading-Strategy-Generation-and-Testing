#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_12_Trend_Filter_1dVolume_Confirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # TRIX on daily close (12-period)
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = pd.Series(ema3).pct_change(periods=1) * 100
    trix_values = trix.values
    trix_signal = (trix > 0).astype(float)  # 1 for bullish, 0 for bearish
    
    # Daily volume filter: current volume > 1.3 * 20-day average
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    vol_filter = volume_1d > (vol_ma20 * 1.3)
    vol_filter_values = vol_filter.values
    
    # Align TRIX signal and volume filter to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for TRIX (12*3=36) + buffer
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(trix_aligned[i]) or np.isnan(vol_filter_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TRIX bullish AND volume confirmation
            if trix_aligned[i] > 0.5 and vol_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX bearish AND volume confirmation
            elif trix_aligned[i] < 0.5 and vol_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns bearish
            if trix_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns bullish
            if trix_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (triple smoothed EMA momentum) on daily timeframe identifies
# intermediate-term trend with less noise. Entry requires TRIX signal AND daily
# volume confirmation to avoid false breakouts. Works in bull markets (TRIX>0 + vol)
# and bear markets (TRIX<0 + vol) by capturing momentum shifts. Volume filter
# ensures participation, reducing whipsaws. Target: 20-35 trades/year for low
# turnover and minimal fee impact. Uses 4h execution for timely entry/exit while
# relying on 1d TRIX for direction and 1d volume for confirmation. Simple, robust
# logic with clear exit conditions to prevent overtrading.