#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation
# Williams Alligator uses smoothed moving averages (SMMA) to identify trend absence/presence
# Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA) - all shifted into future
# Trend present when Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
# 1d EMA50 provides higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>1.5 x 20-period EMA) confirms breakout validity
# Works in bull markets (Alligator uptrend + 1d EMA50 up) and bear markets (Alligator downtrend + 1d EMA50 down)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple SMA
    if len(source) >= length:
        result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (prev_SMMA * (length-1) + current) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Williams Alligator components (all SMMA)
    jaw = smma(close, 13)  # Jaw (13-period SMMA)
    teeth = smma(close, 8)  # Teeth (8-period SMMA)
    lips = smma(close, 5)   # Lips (5-period SMMA)
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator trend
        alligator_uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_downtrend = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend with volume confirmation and 1d uptrend
            if alligator_uptrend and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend with volume confirmation and 1d downtrend
            elif alligator_downtrend and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator trend changes or 1d trend changes to downtrend
            if not alligator_uptrend or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator trend changes or 1d trend changes to uptrend
            if not alligator_downtrend or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals