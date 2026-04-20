#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h EMA Crossover with 1d Trend Filter and Volume Confirmation
# Uses 21/55 EMA crossover on 12h for momentum signals, filtered by 1d EMA50 trend
# Volume > 1.8x 20-period average confirms institutional participation
# Designed for low trade frequency to minimize fee drag while capturing trends
# Target: 12-30 trades per year per symbol (48-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 21 and 55 period EMA on 12h timeframe for crossover signal
    ema21 = pd.Series(prices['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(prices['close'].values).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Calculate volume filter: volume > 1.8x 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(ema21[i]) or np.isnan(ema55[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = prices['close'].values[i] > ema50_1d_aligned[i]
        is_downtrend = prices['close'].values[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        # EMA crossover signals
        ema_cross_up = ema21[i] > ema55[i] and ema21[i-1] <= ema55[i-1]
        ema_cross_down = ema21[i] < ema55[i] and ema21[i-1] >= ema55[i-1]
        
        price = prices['close'].values[i]
        
        if position == 0:
            # Long entry: EMA21 crosses above EMA55 + uptrend + volume
            long_signal = ema_cross_up and is_uptrend and has_volume
            
            # Short entry: EMA21 crosses below EMA55 + downtrend + volume
            short_signal = ema_cross_down and is_downtrend and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA21 crosses below EMA55
            if ema_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA21 crosses above EMA55
            if ema_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA21_55_1dTrendFilter_Volume"
timeframe = "12h"
leverage = 1.0