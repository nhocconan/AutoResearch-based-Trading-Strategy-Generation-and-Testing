#!/usr/bin/env python3
"""
12h Williams Alligator with 1d Trend Filter and Volume Spike.
Long when: 1) Price above Alligator's Jaw (green line), 2) Jaw > Teeth > Lips (bullish alignment), 3) Price > 1d EMA50 (bullish trend), 4) Volume > 2x 20-period average.
Short when: 1) Price below Alligator's Jaw, 2) Jaw < Teeth < Lips (bearish alignment), 3) Price < 1d EMA50 (bearish trend), 4) Volume > 2x 20-period average.
Exit when price crosses the Alligator's Teeth (red line) or trend reverses.
Designed for 12h timeframe: targets 50-150 total trades over 4 years (12-37/year).
Williams Alligator catches trends early; 1d EMA50 filters counter-trend moves; volume confirms breakout strength.
Works in bull (catches trends) and bear (avoids false signals in chop).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price (H+L)/2
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    median_price = (high + low) / 2.0
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA on 12h data, then align to 12h timeframe (no shift for Alligator)
    median_12h = df_12h['median_price'] if 'median_price' in df_12h.columns else (df_12h['high'] + df_12h['low']) / 2.0
    if hasattr(median_12h, 'values'):
        median_12h = median_12h.values
    else:
        median_12h = np.asarray(median_12h)
    
    jaw_raw = smma(median_12h, 13)
    teeth_raw = smma(median_12h, 8)
    lips_raw = smma(median_12h, 5)
    
    # Shift forward: Jaw+8, Teeth+5, Lips+3 (to align with current bar)
    jaw_shifted = np.full_like(jaw_raw, np.nan)
    teeth_shifted = np.full_like(teeth_raw, np.nan)
    lips_shifted = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw_shifted[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth_shifted[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips_shifted[3:] = lips_raw[:-3]
    
    # Align to 12h timeframe (already on 12h bars)
    jaw_12h = jaw_shifted
    teeth_12h = teeth_shifted
    lips_12h = lips_shifted
    
    # Align to current timeframe (12h -> 12h is identity)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (13+8=21 bars), 1d EMA50 (50), volume MA (20)
    start_idx = max(50, 21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Alligator alignment: Jaw > Teeth > Lips (bullish) or Jaw < Teeth < Lips (bearish)
        bullish_alignment = jaw > teeth > lips
        bearish_alignment = jaw < teeth < lips
        
        if position == 0:
            # Long: price above Jaw + bullish alignment + bullish trend + volume spike
            if price > jaw and bullish_alignment and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below Jaw + bearish alignment + bearish trend + volume spike
            elif price < jaw and bearish_alignment and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth (red line) or trend turns bearish
            if price < teeth or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Teeth (red line) or trend turns bullish
            if price > teeth or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0