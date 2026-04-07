#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams Alligator with 1d Trend Filter and Volume Confirmation
# Hypothesis: Williams Alligator (Jaw, Teeth, Lips) identifies trending vs ranging markets.
# In trending markets (JAW > TEETH > LIPS for uptrend, reverse for downtrend), we trade in the direction of the trend.
# Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures trades have institutional participation.
# Works in both bull and bear markets by only taking trades aligned with higher timeframe trend.
# Targets 20-30 trades/year with disciplined entries to avoid overtrading.

name = "6h_williams_alligator_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 6h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward
    # Lips: 5-period SMMA, shifted 3 bars forward
    # Note: SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for indicators
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        # Alligator signals: JAW > TEETH > LIPS = uptrend, reverse = downtrend
        jaw_gt_teeth = jaw[i] > teeth[i]
        teeth_gt_lips = teeth[i] > lips[i]
        jaw_lt_teeth = jaw[i] < teeth[i]
        teeth_lt_lips = teeth[i] < lips[i]
        
        if position == 1:  # Long position
            # Exit: trend changes or price closes below TEETH
            if not (jaw_gt_teeth and teeth_gt_lips) or close[i] < teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: trend changes or price closes above TEETH
            if not (jaw_lt_teeth and teeth_lt_lips) or close[i] > teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Alligator aligned for uptrend + volume confirmation + above 1d EMA50
            if (jaw_gt_teeth and teeth_gt_lips and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: Alligator aligned for downtrend + volume confirmation + below 1d EMA50
            elif (jaw_lt_teeth and teeth_lt_lips and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals