#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with Volume Confirmation
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance.
# Price rejection at these levels with volume confirmation provides high-probability
# reversal entries. Works in both bull and bear markets by fading extremes.
# Uses 12h trend filter to avoid counter-trend trades. Target: 20-40 trades/year.
name = "6h_camarilla_pivot_reversal_v2"
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
    
    # Calculate Camarilla levels from previous day
    # Need daily high, low, close
    # Group by day using date from open_time
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    # Arrays to store daily pivot levels for each bar
    R3 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    R4 = np.full(n, np.nan)
    S4 = np.full(n, np.nan)
    PP = np.full(n, np.nan)
    
    # Calculate for each day
    for d in unique_dates:
        mask = (dates == d)
        if not np.any(mask):
            continue
        
        # Get session high/low/close for the day
        # Use first and last bar of the day for OHLC (simplified)
        day_high = np.max(high[mask])
        day_low = np.min(low[mask])
        day_close = close[mask][-1]  # Last bar of the day
        
        # Camarilla calculations
        range_val = day_high - day_low
        if range_val <= 0:
            continue
            
        PP_val = (day_high + day_low + day_close) / 3
        R3_val = PP_val + (range_val * 1.1 / 4)
        S3_val = PP_val - (range_val * 1.1 / 4)
        R4_val = PP_val + (range_val * 1.1 / 2)
        S4_val = PP_val - (range_val * 1.1 / 2)
        
        # Assign to all bars of this day
        R3[mask] = R3_val
        S3[mask] = S3_val
        R4[mask] = R4_val
        S4[mask] = S4_val
        PP[mask] = PP_val
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Get 12h trend filter (EMA 21)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(PP[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price rejects S3 with volume, bullish 12h trend
        if (low[i] <= S3[i] and close[i] > S3[i] and  # Rejection from S3
            vol_filter[i] and                         # Volume confirmation
            close[i] > ema_12h_aligned[i]):           # Bullish 12h trend
            signals[i] = 0.25
        
        # Short setup: price rejects R3 with volume, bearish 12h trend
        elif (high[i] >= R3[i] and close[i] < R3[i] and  # Rejection from R3
              vol_filter[i] and                          # Volume confirmation
              close[i] < ema_12h_aligned[i]):            # Bearish 12h trend
            signals[i] = -0.25
    
    return signals