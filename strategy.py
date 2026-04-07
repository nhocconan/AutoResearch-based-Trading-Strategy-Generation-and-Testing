#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels with 1-day EMA trend filter and volume confirmation
# Uses daily pivot levels (R3/S3 for mean reversion, R4/S4 for breakouts) from previous day
# In ranging markets: fade extreme levels (R3/S3) with trend filter
# In trending markets: breakout continuation through R4/S4 with volume confirmation
# Designed for low frequency: target 25-40 trades/year to minimize fee drag on 6s timeframe
# Works in bull markets (buy R4 breakouts in uptrend) and bear markets (sell S4 breakdowns in downtrend)

name = "6h_camarilla_pivot_1d_ema_volume_v1"
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
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 6.0)
    s3 = pivot - (range_hl * 1.1 / 6.0)
    r4 = pivot + (range_hl * 1.1 / 2.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Align all levels to 6s timeframe (shifted by 1 day for previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (24-period average = 6 days of 6s data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after volume MA warmup
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime based on price vs pivot
        # If price is within R3-S3 range, consider ranging market
        # If price is outside R4-S4, consider trending market
        in_range = (close[i] >= s3_aligned[i]) and (close[i] <= r3_aligned[i])
        strong_uptrend = close[i] > r4_aligned[i]
        strong_downtrend = close[i] < s4_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions: reversal at S3 or breakdown below S4
            if close[i] <= s3_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit conditions: reversal at R3 or breakout above R4
            if close[i] >= r3_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Mean reversion entries in ranging market: fade extremes with trend filter
            if in_range and vol_confirm:
                # Buy near S3 in uptrend, sell near R3 in downtrend
                if close[i] <= s3_aligned[i] and ema_1d_aligned[i] > close[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= r3_aligned[i] and ema_1d_aligned[i] < close[i]:
                    position = -1
                    signals[i] = -0.25
            # Breakout entries in strong trend with volume confirmation
            elif vol_confirm:
                # Buy R4 breakout in uptrend
                if strong_uptrend and close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Sell S4 breakdown in downtrend
                elif strong_downtrend and close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals