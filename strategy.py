#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA(34) trend filter and volume spike confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# Jaw (13-period, 8-bar shift): Blue line
# Teeth (8-period, 5-bar shift): Red line  
# Lips (5-period, 3-bar shift): Green line
# Trend: Lips > Teeth > Jaw = uptrend (green above red above blue)
#        Lips < Teeth < Jaw = downtrend (green below red below blue)
# Combined with 1d EMA(34) for higher timeframe trend alignment and volume confirmation to reduce false signals.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period
    teeth = smma(close, 8)  # 8-period
    lips = smma(close, 5)   # 5-period
    
    # Apply shifts (Jaw: 8, Teeth: 5, Lips: 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions that would look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (stricter to reduce trades)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams Alligator trend detection
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Lips < Teeth < Jaw
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        
        is_uptrend = (lips_val > teeth_val) and (teeth_val > jaw_val)
        is_downtrend = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator uptrend + price above 1d EMA34 + volume spike
            if is_uptrend and price_above_ema and volume_spike:
                signals[i] = 0.30
                position = 1
            # Short: Alligator downtrend + price below 1d EMA34 + volume spike
            elif is_downtrend and price_below_ema and volume_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Alligator trend changes to downtrend or loses 1d EMA alignment
            if not is_uptrend or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Alligator trend changes to uptrend or loses 1d EMA alignment
            if not is_downtrend or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals