#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + Daily Trend + Volume Confirmation
# Hypothesis: Camarilla levels (R3/S3, R4/S4) from daily pivot provide institutional support/resistance.
# In uptrend (price > daily EMA20): buy at S3 bounce, sell at R4 breakout.
# In downtrend (price < daily EMA20): sell at R3 bounce, buy at S4 breakout.
# Volume spike confirms institutional participation.
# Designed for 6h timeframe with low trade frequency (12-37/year).
# Works in bull via S3 longs + R4 breakouts + volume, in bear via R3 shorts + S4 breakouts + volume.

name = "6h_camarilla_pivot_daily_trend_volume_v1"
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
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point (not used in Camarilla but needed for reference)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = prev_close + range_hl * 1.1 / 2
    r3 = prev_close + range_hl * 1.1 / 4
    s3 = prev_close - range_hl * 1.1 / 4
    s4 = prev_close - range_hl * 1.1 / 2
    
    # Align to 6s timeframe (already shifted by 1 in get_htf_data alignment)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily trend filter: EMA20 of daily close
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 OR daily trend turns bearish
            if close[i] < s3_6h[i] or close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # Short position
            # Exit: price crosses above R3 OR daily trend turns bullish
            if close[i] > r3_6h[i] or close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.28
        else:  # Flat, look for entry
            if vol_ok:
                # Determine trend: price vs daily EMA20
                if close[i] > ema_20_1d_aligned[i]:  # Uptrend
                    # Long: price crosses above S3 (bounce from support)
                    if close[i] > s3_6h[i] and (i == 20 or close[i-1] <= s3_6h[i-1]):
                        position = 1
                        signals[i] = 0.28
                    # Long breakout: price crosses above R4 (continuation)
                    elif close[i] > r4_6h[i] and (i == 20 or close[i-1] <= r4_6h[i-1]):
                        position = 1
                        signals[i] = 0.28
                else:  # Downtrend
                    # Short: price crosses below R3 (bounce from resistance)
                    if close[i] < r3_6h[i] and (i == 20 or close[i-1] >= r3_6h[i-1]):
                        position = -1
                        signals[i] = -0.28
                    # Short breakdown: price crosses below S4 (continuation)
                    elif close[i] < s4_6h[i] and (i == 20 or close[i-1] >= s4_6h[i-1]):
                        position = -1
                        signals[i] = -0.28
    
    return signals