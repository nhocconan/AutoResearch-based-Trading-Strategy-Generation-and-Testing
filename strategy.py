#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
# Long when price > Alligator Jaw (TEETH) in 1d uptrend with volume spike (>2.0x 30-period volume MA).
# Short when price < Alligator Jaw (TEETH) in 1d downtrend with volume spike.
# Alligator (Smoothed MEDIAN price: (H+L+C)/3) identifies trending vs ranging markets.
# 1d EMA50 ensures higher timeframe alignment, avoiding counter-trend trades.
# Volume spike confirms institutional participation. Designed for 6h timeframe to achieve 50-150 total trades over 4 years.

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Alligator calculation (median price)
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Calculate median price: (H+L+C)/3
    median_price = (df_6h['high'].values + df_6h['low'].values + df_6h['close'].values) / 3.0
    
    # Alligator components: Smoothed median price with different periods
    # Jaw (blue): 13-period smoothed, 8 bars ahead
    jaw_raw = pd.Series(median_price).ewm(span=13, adjust=False).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth (red): 8-period smoothed, 5 bars ahead
    teeth_raw = pd.Series(median_price).ewm(span=8, adjust=False).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips (green): 5-period smoothed, 3 bars ahead
    lips_raw = pd.Series(median_price).ewm(span=5, adjust=False).mean().values
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid
    
    # Align Alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (30-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        # Alligator alignment: Mouth open (trending) when Lips > Teeth > Jaw (up) or Lips < Teeth < Jaw (down)
        mouth_open_up = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        mouth_open_down = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long: price > Teeth AND 1d uptrend AND volume spike AND mouth open up
            if close_val > teeth_aligned[i] and trend_up and vol_spike and mouth_open_up:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price < Teeth AND 1d downtrend AND volume spike AND mouth open down
            elif close_val < teeth_aligned[i] and trend_down and vol_spike and mouth_open_down:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price < Jaw (Alligator sleeping - trend over)
            if close_val < jaw_aligned[i]:
                exit_signal = True
            # Exit: 1d trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            # Exit: Mouth closes (Lips < Teeth - losing momentum)
            elif lips_aligned[i] < teeth_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price > Jaw (Alligator sleeping - trend over)
            if close_val > jaw_aligned[i]:
                exit_signal = True
            # Exit: 1d trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            # Exit: Mouth closes (Lips > Teeth - losing momentum)
            elif lips_aligned[i] > teeth_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals