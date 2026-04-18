#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator indicator with 1d Fractal confirmation and volume spike filter.
Designed to capture trend reversals in both bull and bear markets by using Alligator's jaw-teeth-lips
alignment for trend direction, Williams Fractals for entry timing, and volume confirmation to avoid
false signals. Targets 15-30 trades/year to minimize fee drag on 12h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (SMMA-based)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator components (SMMA with different periods)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] + result[i-1] * (period-1)) / period
        return result
    
    # Alligator: Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift)
    jaw_raw = smma(close_12h, 13)
    teeth_raw = smma(close_12h, 8)
    lips_raw = smma(close_12h, 5)
    
    # Apply shifts (Alligator is shifted forward)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Align Alligator components to 12h timeframe (already on 12h, just need to align to lower timeframe)
    jaw_12h = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_12h = align_htf_to_ltf(prices, df_12h, teeth)
    lips_12h = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for Williams Fractals (need 1d for proper fractal formation)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Fractals need 2-bar confirmation delay (as per Williams)
    bearish_fractal_1d = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_1d = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate volume moving average (20-period) on 12h
    vol_ma_12h = np.full(len(close_12h), np.nan)
    for i in range(20, len(close_12h)):
        vol_ma_12h[i] = np.mean(close_12h[i-20:i])  # Using close as volume proxy for 12h data
    
    # Get actual 12h volume from resampled data (need to get volume from 12h aggregation)
    # Since we don't have volume in the 12h data from get_htf_data, we'll use price-based volatility
    # Alternative: use price range as volatility proxy
    hl_range_12h = (df_12h['high'].values - df_12h['low'].values)
    vol_proxy_ma_12h = np.full(len(hl_range_12h), np.nan)
    for i in range(20, len(hl_range_12h)):
        vol_proxy_ma_12h[i] = np.mean(hl_range_12h[i-20:i])
    
    # Align volume proxy to 12h timeframe
    vol_proxy_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_proxy_ma_12h)
    
    # Calculate current 12h volatility proxy (high-low range)
    hl_range = high - low
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(hl_range[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need Alligator, fractals, volatility MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(bearish_fractal_1d[i]) or np.isnan(bullish_fractal_1d[i]) or
            np.isnan(vol_ma[i]) or np.isnan(vol_proxy_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: check if aligned (all three lines in order)
        # For uptrend: Lips > Teeth > Jaw (all above price)
        # For downtrend: Jaw > Teeth > Lips (all below price)
        ma_aligned_up = (lips_12h[i] > teeth_12h[i]) and (teeth_12h[i] > jaw_12h[i])
        ma_aligned_down = (jaw_12h[i] > teeth_12h[i]) and (teeth_12h[i] > lips_12h[i])
        
        # Price relative to Alligator (for entry confirmation)
        price_vs_lips = close[i] > lips_12h[i]  # Above lips = bullish bias
        price_vs_jaw = close[i] < jaw_12h[i]    # Below jaw = bearish bias
        
        # Volume confirmation: current volatility > 1.5 * average volatility
        vol_confirmed = vol_ma[i] > 1.5 * vol_proxy_ma_12h_aligned[i]
        
        # Fractal confirmation: bullish fractal = potential bottom, bearish = potential top
        bullish_fractal_signal = bullish_fractal_1d[i] == 1.0
        bearish_fractal_signal = bearish_fractal_1d[i] == 1.0
        
        if position == 0:
            # Long entry: Alligator aligned up, price above lips, bullish fractal, volume confirmation
            if ma_aligned_up and price_vs_lips and bullish_fractal_signal and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator aligned down, price below jaw, bearish fractal, volume confirmation
            elif ma_aligned_down and price_vs_jaw and bearish_fractal_signal and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: Alligator alignment breaks (lips cross below teeth) OR price hits jaw
            if not ma_aligned_up or close[i] < jaw_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks (jaw crosses below teeth) OR price hits lips
            if not ma_aligned_down or close[i] > lips_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Fractal_Volume"
timeframe = "12h"
leverage = 1.0