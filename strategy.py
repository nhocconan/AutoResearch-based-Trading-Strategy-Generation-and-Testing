#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# Bullish: Bull Power > 0 and Bear Power < 0 (strong uptrend)
# Bearish: Bear Power < 0 and Bull Power < 0 (strong downtrend)
# Uses 1d EMA34 for trend filter to avoid whipsaws in sideways markets
# Volume > 1.5x 20-period average confirms conviction
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray (using 6h data)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # 34-period EMA on daily close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Bull Power > 0 AND Bear Power < 0 (strong uptrend) 
        # AND price above daily EMA34 (uptrend filter) AND volume confirmation
        if (bull_power[i] > 0 and bear_power[i] < 0 and 
            close[i] > ema34_1d_aligned[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Bear Power < 0 AND Bull Power < 0 (strong downtrend)
        # AND price below daily EMA34 (downtrend filter) AND volume confirmation
        elif (bear_power[i] < 0 and bull_power[i] < 0 and 
              close[i] < ema34_1d_aligned[i] and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0