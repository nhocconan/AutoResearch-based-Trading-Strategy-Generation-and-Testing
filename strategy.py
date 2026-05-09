#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume spike confirmation.
# Alligator uses smoothed moving averages (Jaw=13, Teeth=8, Lips=5) to identify trends.
# In uptrend: Lips > Teeth > Jaw; in downtrend: Lips < Teeth < Jaw.
# Combined with 1d EMA50 for trend direction and volume spike (>1.8x average) for confirmation.
# Designed to catch strong trends in both bull/bear markets while avoiding chop.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple moving average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)   # Jaw (13-period)
    teeth = smma(close, 8)  # Teeth (8-period)
    lips = smma(close, 5)   # Lips (5-period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and Alligator
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        # Alligator signals:
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Lips < Teeth < Jaw
        is_uptrend = lips_val > teeth_val and teeth_val > jaw_val
        is_downtrend = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Enter long: Uptrend AND price > 1d EMA50 (uptrend) AND volume > 1.8x average
            if is_uptrend and close[i] > ema_1d and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend AND price < 1d EMA50 (downtrend) AND volume > 1.8x average
            elif is_downtrend and close[i] < ema_1d and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend breaks (not uptrend) OR price < 1d EMA50
            if not is_uptrend or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend breaks (not downtrend) OR price > 1d EMA50
            if not is_downtrend or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals