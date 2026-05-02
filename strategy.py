#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 12h EMA50 Trend + Volume Confirmation
# Elder Ray measures bull/bear power relative to EMA13. In strong trends (12h EMA50), 
# we take pullbacks: long when Bull Power > 0 and Bear Power < 0 in uptrend, 
# short when Bear Power < 0 and Bull Power > 0 in downtrend. Volume spike confirms momentum.
# Works in both bull and bear markets by aligning with higher-timeframe trend.
# Target: 50-150 trades over 4 years (12-37/year) on 6h.

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    if len(close) < 13:
        return np.zeros(n)
    
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 1.8x 24-period average (~6 days for 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for 12h EMA50 and EMA13)
    start_idx = max(50, 13)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Uptrend (price > 12h EMA50) AND Bull Power > 0 AND Bear Power < 0 AND volume spike
            if (close[i] > ema_50_12h_aligned[i] and 
                bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Downtrend (price < 12h EMA50) AND Bear Power < 0 AND Bull Power > 0 AND volume spike
            elif (close[i] < ema_50_12h_aligned[i] and 
                  bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Trend change (price < 12h EMA50) OR Bull Power <= 0 (loss of bullish momentum)
            if close[i] < ema_50_12h_aligned[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Trend change (price > 12h EMA50) OR Bear Power >= 0 (loss of bearish momentum)
            if close[i] > ema_50_12h_aligned[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals