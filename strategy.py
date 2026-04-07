#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Weekly Camarilla Pivot with Volume and ATR Filter
# Hypothesis: Weekly Camarilla levels (R3/S3 and R4/S4) act as strong institutional barriers.
# Price breaking above R4 with volume indicates bullish continuation; breaking below S4 indicates bearish continuation.
# Price bouncing off R3/S3 with volume indicates mean reversion.
# Works in both bull and bear markets: In bull, breaks above R4 continue up; breaks below S3 get bought.
# In bear, breaks below S4 continue down; breaks above R3 get sold.
# Volume filter ensures only institutional participation triggers entries.
# ATR filter avoids choppy markets. Target: 15-40 trades/year (60-160 over 4 years).

name = "4h_weekly_camarilla_pivot_volume_atr_v1"
timeframe = "4h"
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
    
    # Get weekly data for Camarilla calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly data (previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate weekly Camarilla pivot points
    weekly_range = prev_weekly_high - prev_weekly_low
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r3 = weekly_pivot + (weekly_range * 1.1 / 2)
    weekly_s3 = weekly_pivot - (weekly_range * 1.1 / 2)
    weekly_r4 = weekly_pivot + (weekly_range * 1.1)
    weekly_s4 = weekly_pivot - (weekly_range * 1.1)
    
    # Align to 4h timeframe (use previous week's levels)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR filter: avoid choppy markets (ATR < 50-period average)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr < atr_ma  # Only trade when volatility is below average (avoid chop)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to S3 or volume drops or volatility spikes
            if (close[i] <= weekly_s3_aligned[i] or not vol_filter[i] or not atr_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to R3 or volume drops or volatility spikes
            if (close[i] >= weekly_r3_aligned[i] or not vol_filter[i] or not atr_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R4 with volume and low volatility
            if ((high[i] > weekly_r4_aligned[i]) and 
                (close[i] > weekly_r4_aligned[i]) and 
                vol_filter[i] and atr_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume and low volatility
            elif ((low[i] < weekly_s4_aligned[i]) and 
                  (close[i] < weekly_s4_aligned[i]) and 
                  vol_filter[i] and atr_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals