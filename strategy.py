#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (JAW=13, TEETH=8, LIPS=5 SMMA) to identify trend absence/presence
# 1d EMA50 ensures alignment with long-term trend to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Works in bull markets (trend following) and bear markets (mean reversion during Alligator sleep)

name = "12h_Williams_Alligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 20-period average (~10 days for 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Williams Alligator calculation (SMMA = Smoothed Moving Average)
    def smma(arr, period):
        """Smoothed Moving Average - similar to RMA/Wilder's smoothing"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and EMA50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator sleeping: jaws, teeth, lips intertwined (no trend)
            alligator_sleeping = (abs(jaw[i] - teeth[i]) < (close[i] * 0.001) and 
                                abs(teeth[i] - lips[i]) < (close[i] * 0.001) and
                                abs(lips[i] - jaw[i]) < (close[i] * 0.001))
            
            # Alligator awakening: lines diverging with trend
            bullish_awakening = (lips[i] > teeth[i] > jaw[i])  # Green > Red > Blue
            bearish_awakening = (jaw[i] > teeth[i] > lips[i])  # Blue > Red > Green
            
            # Long entry: Bullish awakening AND price > 1d EMA50 AND volume spike
            if (bullish_awakening and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish awakening AND price < 1d EMA50 AND volume spike
            elif (bearish_awakening and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator sleeping again (trend exhaustion) OR price < 1d EMA50 (trend change)
            alligator_sleeping = (abs(jaw[i] - teeth[i]) < (close[i] * 0.001) and 
                                abs(teeth[i] - lips[i]) < (close[i] * 0.001) and
                                abs(lips[i] - jaw[i]) < (close[i] * 0.001))
            if alligator_sleeping or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator sleeping again (trend exhaustion) OR price > 1d EMA50 (trend change)
            alligator_sleeping = (abs(jaw[i] - teeth[i]) < (close[i] * 0.001) and 
                                abs(teeth[i] - lips[i]) < (close[i] * 0.001) and
                                abs(lips[i] - jaw[i]) < (close[i] * 0.001))
            if alligator_sleeping or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals