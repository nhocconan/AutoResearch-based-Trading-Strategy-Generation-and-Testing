#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1week trend filter and volume confirmation
# Williams Alligator uses smoothed moving averages (Jaw, Teeth, Lips) to identify
# trend presence and direction. In strong trends, the three lines are aligned and
# separated (alligator "opens mouth"). In ranging markets, they intertwine.
# We add 1week trend filter (price > EMA50) to ensure we trade only in strong
# trends, and volume confirmation to avoid false signals.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams Alligator: three SMMA lines
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA (Smoothed Moving Average) is like EMA but with alpha = 1/period
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        res = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        res[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    # Calculate Alligator lines
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift the lines forward (Jaw+8, Teeth+5, Lips+3)
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Align to 6s timeframe (wait for 6s bar close)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw_shifted)  # same timeframe
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, prices, lips_shifted)
    
    # 1week EMA trend filter (50-period)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (13+8=21), EMA (50), volume MA (20)
    start_idx = max(21, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Alligator conditions:
        # Mouth open (trending): Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # 1week trend filter
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: bullish alignment + volume + bullish weekly trend
            if bullish_aligned and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: bearish alignment + volume + bearish weekly trend
            elif bearish_aligned and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: alignment breaks or trend turns bearish
            if not bullish_aligned or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: alignment breaks or trend turns bullish
            if not bearish_aligned or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsAlligator_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0