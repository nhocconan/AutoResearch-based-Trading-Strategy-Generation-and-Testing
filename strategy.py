#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA8 filter and volume confirmation
# Williams Alligator (Jaw=TEETH=LIPS) uses SMAs to identify trends:
# Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
# In uptrend: Lips > Teeth > Jaw; in downtrend: Jaw > Teeth > Lips
# Combined with 1d EMA8 trend filter and volume confirmation to avoid false signals
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "12h_WilliamsAlligator_1dEMA8_Volume"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev*(period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA8 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 8:
        return np.zeros(n)
    
    ema_8_1d = pd.Series(df_1d['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_8_1d)
    
    # Williams Alligator components on 12h
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # shift right by 8
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_8_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + above 1d EMA8 + volume confirmation
            if (lips[i] > teeth[i] > jaw[i] and 
                close[i] > ema_8_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) + below 1d EMA8 + volume confirmation
            elif (jaw[i] > teeth[i] > lips[i] and 
                  close[i] < ema_8_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if bearish alignment (Jaw > Teeth > Lips) or breaks below 1d EMA8
            if (jaw[i] > teeth[i] > lips[i]) or (close[i] < ema_8_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if bullish alignment (Lips > Teeth > Jaw) or breaks above 1d EMA8
            if (lips[i] > teeth[i] > jaw[i]) or (close[i] > ema_8_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals