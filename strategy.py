#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when Alligator jaws (SMMA13) > teeth (SMMA8) > lips (SMMA5) on 12h, 1d EMA50 rising, volume > 1.5x average
# Short when Alligator jaws < teeth < lips, 1d EMA50 falling, volume > 1.5x average
# Uses 12h for entry timing, 1d for trend filter to avoid whipsaws in choppy markets
# Targets 50-150 total trades over 4 years (12-37/year) for low fee drag and high win rate

name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan)
    result = np.full_like(data, np.nan, dtype=np.float64)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
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
    
    # Get 12h data for Alligator (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Alligator components on 12h median price (typical price)
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    jaws = smma(typical_price_12h, 13)  # SMMA(13)
    teeth = smma(typical_price_12h, 8)   # SMMA(8)
    lips = smma(typical_price_12h, 5)    # SMMA(5)
    
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # warmup for Alligator jaws (longest period)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaws_aligned[i]
        tooth_val = teeth_aligned[i]
        lip_val = lips_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: Alligator aligned (jaws > teeth > lips), 1d uptrend, volume spike
            if jaw_val > tooth_val and tooth_val > lip_val and ema50_1d_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator aligned (jaws < teeth < lips), 1d downtrend, volume spike
            elif jaw_val < tooth_val and tooth_val < lip_val and ema50_1d_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator misaligned (jaws <= teeth) or 1d trend down
            if jaw_val <= tooth_val or ema50_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator misaligned (jaws >= teeth) or 1d trend up
            if jaw_val >= tooth_val or ema50_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals