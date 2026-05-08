#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# We go long when Lips > Teeth > Jaw (bullish alignment) and short when Lips < Teeth < Jaw (bearish alignment),
# confirmed by 1w EMA(34) trend direction and volume spike.
# Designed for low trade frequency in both bull and bear markets.
# Target: 30-100 total trades over 4 years = 7-25/year (1d timeframe)

name = "1d_WilliamsAlligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
    smma_vals = np.full_like(data, np.nan, dtype=float)
    if len(data) >= period:
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(data)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + data[i]) / period
    return smma_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Williams Alligator components on 1d data
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift the averages as per Alligator definition
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) >= 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) >= 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) >= 3:
        lips_shifted[3:] = lips[:-3]
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + uptrend + volume spike
            if (lips_val > teeth_val > jaw_val and 
                close[i] > ema34_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + downtrend + volume spike
            elif (lips_val < teeth_val < jaw_val and 
                  close[i] < ema34_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bullish alignment breaks OR price breaks below trend
            if not (lips_val > teeth_val > jaw_val) or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bearish alignment breaks OR price breaks above trend
            if not (lips_val < teeth_val < jaw_val) or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals