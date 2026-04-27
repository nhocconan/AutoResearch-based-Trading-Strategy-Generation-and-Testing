#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 12h trend filter and volume confirmation
# Alligator lines act as dynamic support/resistance; when price trades outside the mouth (all lines),
# it indicates a trend. Combined with 12h EMA trend and volume spike, this captures strong moves
# while avoiding whipsaws in sideways markets. Works in bull/bear by filtering breakout direction
# with 12h EMA trend. Target: 50-150 total trades over 4 years (~12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: SMoothed Moving Average (SMMA)
    # Jaw (13-period, 8 bars ahead), Teeth (8-period, 5 bars ahead), Lips (5-period, 3 bars ahead)
    def smma(arr, period):
        """Smoothed Moving Average"""
        sma = np.full(len(arr), np.nan)
        if len(arr) < period:
            return sma
        sma[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw_raw = smma(close_12h, 13)
    teeth_raw = smma(close_12h, 8)
    lips_raw = smma(close_12h, 5)
    
    # Shift jaw forward 8 bars, teeth forward 5 bars, lips forward 3 bars
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # 12h EMA trend filter (50-period)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 2.0 x 24-period average (4 days of 6h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h data (13 bars), Alligator (max shift 8), EMA (50), volume MA (24)
    start_idx = max(13, 8, 50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from 12h EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        # Mouth condition: all three lines ordered (Jaw > Teeth > Lips for down, Lips > Teeth > Jaw for up)
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        mouth_open_up = lips_val > teeth_val > jaw_val  # Bullish alignment
        mouth_open_down = jaw_val > teeth_val > lips_val  # Bearish alignment
        
        if position == 0:
            # Long: price above all lines (Lips highest) with volume and bullish trend
            if price > lips_val and mouth_open_up and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price below all lines (Jaw lowest) with volume and bearish trend
            elif price < jaw_val and mouth_open_down and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth or trend turns bearish
            if price < teeth_val or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Teeth or trend turns bullish
            if price > teeth_val or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Williams_Alligator_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0