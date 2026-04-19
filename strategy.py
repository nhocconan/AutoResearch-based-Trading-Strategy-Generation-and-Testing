#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + daily VWAP + volume confirmation.
# Williams Alligator (Jaw/Teeth/Lips) identifies trends via SMAs with future offsets.
# Price > Lips and Lips > Teeth > Jaw = uptrend; Price < Lips and Lips < Teeth < Jaw = downtrend.
# Daily VWAP acts as dynamic support/resistance: price above VWAP supports longs, below supports shorts.
# Volume confirmation ensures breakouts have conviction.
# Works in bull/bear markets: Alligator catches trends, VWAP filters mean reversion, volume avoids false signals.
# Target: 20-40 trades/year per symbol.
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
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate VWAP on daily
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / vwap_den
    
    # Williams Alligator on 12h: SMAs with future offsets
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    jaw_period, jaw_shift = 13, 8
    teeth_period, teeth_shift = 8, 5
    lips_period, lips_shift = 5, 3
    
    # Calculate SMMA (smoothed moving average) = EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        alpha = 1.0 / period
        result = np.full_like(arr, np.nan, dtype=float)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] * alpha) + (result[i-1] * (1 - alpha))
        return result
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # Shift forward (to avoid look-ahead, we use values that would be known at bar close)
    jaw_shifted = np.roll(jaw, -jaw_shift)
    teeth_shifted = np.roll(teeth, -teeth_shift)
    lips_shifted = np.roll(lips, -lips_shift)
    # Set shifted-out values to NaN
    jaw_shifted[-jaw_shift:] = np.nan
    teeth_shifted[-teeth_shift:] = np.nan
    lips_shifted[-lips_shift:] = np.nan
    
    # Align 1d VWAP to 12h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        vwap_val = vwap_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Williams Alligator conditions
        # Uptrend: Lips > Teeth > Jaw and price > Lips
        # Downtrend: Lips < Teeth < Jaw and price < Lips
        lips_above_teeth = lips_val > teeth_val
        teeth_above_jaw = teeth_val > jaw_val
        lips_below_teeth = lips_val < teeth_val
        teeth_below_jaw = teeth_val < jaw_val
        
        uptrend = lips_above_teeth and teeth_above_jaw and (price > lips_val)
        downtrend = lips_below_teeth and teeth_below_jaw and (price < lips_val)
        
        # VWAP filter: price above VWAP longs bias, below shorts bias
        price_above_vwap = price > vwap_val
        price_below_vwap = price < vwap_val
        
        # Volume confirmation
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Enter long: uptrend + price above VWAP + volume
            if uptrend and price_above_vwap and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + price below VWAP + volume
            elif downtrend and price_below_vwap and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when trend changes or price crosses below VWAP
            if not uptrend or price < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when trend changes or price crosses above VWAP
            if not downtrend or price > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals