#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Williams Alligator (3 SMAs) with Elder Ray power and volume spike.
- Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
- Enter long when Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + volume > 1.5x 20-period volume MA
- Enter short when Lips < Teeth < Jaw (bearish alignment) + Bear Power > 0 + volume > 1.5x 20-period volume MA
- Exit when Alligator alignment reverses (Lips crosses Teeth)
- Fixed position size 0.25 to manage drawdown
- Uses Elder Ray for trend confirmation and volume for momentum confirmation
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA)"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.full_like(series, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(series[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for indicator calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA
    jaw = smma(close_4h, 13)
    teeth = smma(close_4h, 8)
    lips = smma(close_4h, 5)
    
    # Elder Ray: EMA13 for Bull/Bear Power
    ema_13 = pd.Series(close_4h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_4h - ema_13
    bear_power = ema_13 - low_4h
    
    # Align all indicators to 4h timeframe (they're already on 4h, but align for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_4h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_4h, bear_power)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume_4h[i] if i < len(volume_4h) else volume_ma_aligned[i]  # fallback
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips_val > teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Look for entry signals
            # Long: bullish alignment + bull power positive + volume spike
            if bullish_alignment and bull_power_val > 0 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + bear power positive + volume spike
            elif bearish_alignment and bear_power_val > 0 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: exit when bullish alignment breaks (lips crosses below teeth)
            if lips_val <= teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit when bearish alignment breaks (lips crosses above teeth)
            if lips_val >= teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0