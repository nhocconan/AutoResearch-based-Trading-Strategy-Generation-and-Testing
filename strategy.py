#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + 1d EMA34 Trend Filter
# Williams Alligator identifies trend direction and strength using smoothed moving averages.
# In trending markets (Alligator awake), price stays aligned with the jaw/teeth/lips.
# Volume spike confirms institutional participation in the trend.
# 1d EMA34 filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Works in bull markets (price above Alligator) and bear markets (price below Alligator).
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
name = "12h_WilliamsAlligator_Volume_1dEMA34"
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
    
    # Get 12h data for Williams Alligator (smoothed medians)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Williams Alligator components: Jaw (13), Teeth (8), Lips (5)
    # Using SMMA (Smoothed Moving Average) with specific smoothing
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_12h, 13)  # Blue line
    teeth = smma(median_12h, 8)  # Red line
    lips = smma(median_12h, 5)   # Green line
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 12-period average volume (1.5 days on 12h chart)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (2.0 * vol_ma_12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_12[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_34_1d_aligned[i]
        
        if position == 0:
            # Alligator awake: lips > teeth > jaw (uptrend) OR lips < teeth < jaw (downtrend)
            # Long: Price above lips AND lips > teeth > jaw AND price above EMA34 AND volume spike
            if (close_val > lips_val and lips_val > teeth_val and teeth_val > jaw_val and
                close_val > ema_val and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below lips AND lips < teeth < jaw AND price below EMA34 AND volume spike
            elif (close_val < lips_val and lips_val < teeth_val and teeth_val < jaw_val and
                  close_val < ema_val and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price below teeth (trend weakening) or below EMA34
            if close_val < teeth_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price above teeth (trend weakening) or above EMA34
            if close_val > teeth_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals