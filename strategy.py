#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla pivot with 1d trend filter and volume confirmation
# Uses daily Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout)
# 1d EMA50 determines trend direction: only long above, short below
# Volume confirmation: current 6h volume > 20-period 1d average volume
# Target: 12-30 trades/year (50-120 over 4 years) to minimize fee drag
# Works in bull via breakout continuation, in bear via mean reversion at S3/R3
name = "6h_camarilla_pivot_1d_trend_volume_v1"
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
    
    # Get daily data for Camarilla pivot, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    # Where C, H, L are from previous day
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First day has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_base = prev_close
    rang = prev_high - prev_low
    r4 = camarilla_base + (rang * 1.1 / 2)
    r3 = camarilla_base + (rang * 1.1 / 4)
    s3 = camarilla_base - (rang * 1.1 / 4)
    s4 = camarilla_base - (rang * 1.1 / 2)
    
    # Align daily data to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 20-period average volume from daily data for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if required data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches S3 (mean reversion) OR breaks above R4 (trailing stop)
            if close[i] <= s3_aligned[i] or close[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches R3 (mean reversion) OR breaks below S4 (trailing stop)
            if close[i] >= r3_aligned[i] or close[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Determine trend direction from daily EMA50
            uptrend = close[i] > ema50_1d_aligned[i]
            downtrend = close[i] < ema50_1d_aligned[i]
            
            # Enter long: in uptrend, price breaks above R4 with volume (breakout continuation)
            #           OR in any trend, price touches S3 with volume (mean reversion)
            if ((uptrend and close[i] > r4_aligned[i]) or 
                (close[i] <= s3_aligned[i])) and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: in downtrend, price breaks below S4 with volume (breakout continuation)
            #            OR in any trend, price touches R3 with volume (mean reversion)
            elif ((downtrend and close[i] < s4_aligned[i]) or 
                  (close[i] >= r3_aligned[i])) and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals