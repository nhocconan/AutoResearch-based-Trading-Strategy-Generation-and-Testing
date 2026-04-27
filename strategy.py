#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 1-day trend filter and volume spike filter.
- Long when Alligator bullish (JAW < TEETH < LIPS) + price > 1d EMA50 (uptrend) + volume spike (>1.8x avg)
- Short when Alligator bearish (JAW > TEETH > LIPS) + price < 1d EMA50 (downtrend) + volume spike (>1.8x avg)
- Exit when Alligator cross reverses (JAW/TEETH/LIPS not in order) or trend reverses
- Williams Alligator catches trends early with smoothing; volume confirmation reduces false signals
- Uses 1d for trend direction (fewer signals), 4h for precise entry timing
- Target: 20-40 trades/year (80-160 over 4 years) to minimize fee drag
- Works in bull/bear via trend filter - only trades with higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator: SMMA (Smoothed Moving Average)
    # JAW: 13-period SMMA, shifted 8 bars forward
    # TEETH: 8-period SMMA, shifted 5 bars forward  
    # LIPS: 5-period SMMA, shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - also called Wilder's smoothing"""
        result = np.full(len(arr), np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Shift forward: JAW+8, TEETH+5, LIPS+3
    jaw = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    
    for i in range(8, n):
        if not np.isnan(jaw_raw[i-8]):
            jaw[i] = jaw_raw[i-8]
    for i in range(5, n):
        if not np.isnan(teeth_raw[i-5]):
            teeth[i] = teeth_raw[i-5]
    for i in range(3, n):
        if not np.isnan(lips_raw[i-3]):
            lips[i] = lips_raw[i-3]
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 1.8 * 30-period average (higher threshold for fewer trades)
    vol_ma_30 = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_30[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for Alligator and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Alligator bullish: JAW < TEETH < LIPS (green alignment)
            # Alligator bearish: JAW > TEETH > LIPS (red alignment)
            bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Long entry: Alligator bullish + above 1d EMA50 + volume spike
            if (bullish_alignment and close[i] > ema50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish + below 1d EMA50 + volume spike
            elif (bearish_alignment and close[i] < ema50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Alligator alignment breaks (not bullish) OR below 1d EMA50 (trend change)
            bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            if (not bullish_alignment or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks (not bearish) OR above 1d EMA50 (trend change)
            bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            if (not bearish_alignment or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0