#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray with 1d EMA50 trend filter and volume confirmation
# Uses Alligator (JAWS/TEETH/LIPS) to identify trend direction and avoid ranging markets
# Elder Ray (Bull/Bear Power) measures trend strength relative to EMA13
# 1d EMA50 ensures alignment with higher timeframe trend to reduce false signals
# Volume spike (1.8x 20-period average) confirms institutional participation
# Designed for both bull and bear markets by following 1d trend
# Target: 80-180 total trades over 4 years (20-45/year)

name = "4h_WilliamsAlligator_ElderRay_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams Alligator on 4h: SMAs with specific periods
    # JAWS: 13-period SMMA shifted 8 bars ahead
    # TEETH: 8-period SMMA shifted 5 bars ahead  
    # LIPS: 5-period SMMA shifted 3 bars ahead
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_PRICE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaws = smma(high, 13)  # Using high for JAWS as per traditional Alligator
    teeth = smma(low, 8)   # Using low for TEETH
    lips = smma(close, 5)  # Using close for LIPS
    
    # Shift the lines as per Alligator definition
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that rolled from beginning
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align to 4h timeframe (already on 4h, but ensuring proper alignment)
    jaws_aligned = jaws_shifted
    teeth_aligned = teeth_shifted
    lips_aligned = lips_shifted
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaws = uptrend
            # Lips < Teeth < Jaws = downtrend
            # Elder Ray confirmation + volume spike
            
            # Long: Bullish alignment + positive Bull Power + price > 1d EMA50 + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaws_aligned[i] and 
                bull_power[i] > 0 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + positive Bear Power + price < 1d EMA50 + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i] and 
                  bear_power[i] > 0 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR Elder Ray turns negative
            if (lips_aligned[i] < teeth_aligned[i] or 
                bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR Elder Ray turns negative
            if (lips_aligned[i] > teeth_aligned[i] or 
                bear_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals