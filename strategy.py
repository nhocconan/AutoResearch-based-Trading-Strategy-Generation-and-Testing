#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot with 1d Trend Filter and Volume Confirmation
# Hypothesis: Camarilla levels provide institutional support/resistance; 1d trend filters direction; volume confirms breakout/breakdown validity.
# Works in bull via R4 breakouts in uptrend, in bear via S4 breakdowns in downtrend. Volume avoids false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
name = "6h_camarilla_1d_trend_volume_v1"
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
    open_price = prices['open'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # H2 = close + 1.0 * (high - low)
    # L2 = close - 1.0 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # Pivot = (high + low + close) / 3
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    hl_range = prev_high - prev_low
    r4 = prev_close + 1.5 * hl_range
    r3 = prev_close + 1.25 * hl_range
    s3 = prev_close - 1.25 * hl_range
    s4 = prev_close - 1.5 * hl_range
    pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align to 6h timeframe (shifted by 1 day for look-ahead avoidance)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Daily trend filter: 20-period EMA
    ema_20 = df_1d['close'].ewm(span=20, min_periods=20).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Daily volume average for confirmation
    vol_ma_20 = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below R3 (take profit) or below S4 (stop)
            if close[i] < r3_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above S3 (take profit) or above R4 (stop)
            if close[i] > s3_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above R4 with uptrend and volume confirmation
            if (close[i] > r4_aligned[i] and 
                close[i] > ema_aligned[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S4 with downtrend and volume confirmation
            elif (close[i] < s4_aligned[i] and 
                  close[i] < ema_aligned[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals