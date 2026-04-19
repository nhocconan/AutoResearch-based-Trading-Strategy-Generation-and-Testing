#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with daily VWAP filter and volume confirmation.
# Long when price > Alligator Jaw (teeth) and price > daily VWAP and volume > 1.5x 12h average volume.
# Short when price < Alligator Jaw (teeth) and price < daily VWAP and volume > 1.5x 12h average volume.
# Exit when price crosses back below/above Alligator Jaw (teeth).
# Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
# Uses Jaw as the primary trend filter. Aims for fewer, high-quality trades in both bull and bear markets.
# Target: 12-37 trades/year per symbol to stay within frequency limits.
name = "12h_WilliamsAlligator_DailyVWAP_Volume"
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
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily VWAP (typical price * volume) / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Align daily VWAP to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # Calculate Williams Alligator on close prices
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = smma(close, 13)
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan  # First 8 values invalid after shift
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = smma(close, 8)
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan  # First 5 values invalid after shift
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = smma(close, 5)
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan  # First 3 values invalid after shift
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA and indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_aligned[i]
        jaw = jaw_shifted[i]
        teeth = teeth_shifted[i]
        lips = lips_shifted[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        # Use Jaw as primary trend filter with lips/teeth for confirmation
        if position == 0:
            # Long entry: price > Jaw AND price > VWAP AND volume confirmation
            if price > jaw and price > vwap and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Jaw AND price < VWAP AND volume confirmation
            elif price < jaw and price < vwap and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Jaw
            if price < jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Jaw
            if price > jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals