#!/usr/bin/env python3
"""
12h_TRIX_Volume_Spike_1dTrend_Filter_v1
Hypothesis: Use TRIX (15-period) on 12h timeframe for momentum with 1d trend filter (close > 50 EMA) and volume spike (>1.5x 20-period average) to confirm. TRIX captures momentum changes while volume spike ensures institutional participation. Works in bull (momentum continuation) and bear (momentum reversal) markets by going long when TRIX turns up with volume, short when TRIX turns down with volume. Designed for 12h to limit trades (~20-50/year) and avoid fee drag.
"""

name = "12h_TRIX_Volume_Spike_1dTrend_Filter_v1"
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50 EMA for 1d trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX on 12h price
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period rate of change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # percentage change
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(trix.iloc[i]) or 
            np.isnan(volume_spike.iloc[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX turning up (positive and rising) with volume spike and above 1d EMA50
            if trix.iloc[i] > 0 and trix.iloc[i] > trix.iloc[i-1] and volume_spike.iloc[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX turning down (negative and falling) with volume spike and below 1d EMA50
            elif trix.iloc[i] < 0 and trix.iloc[i] < trix.iloc[i-1] and volume_spike.iloc[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns down or breaks below 1d EMA50
            if trix.iloc[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns up or breaks above 1d EMA50
            if trix.iloc[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals