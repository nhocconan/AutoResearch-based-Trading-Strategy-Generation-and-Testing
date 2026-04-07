#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + Volume + Trend Filter
# Hypothesis: Camarilla levels provide precise support/resistance on 1d timeframe.
# Fade at R3/S3 (mean reversion), breakout at R4/S4 (trend continuation).
# Volume confirms breakout strength. 12h EMA filters trend direction.
# Works in bull/bear by adapting to price action at key levels.
# Target: 12-30 trades/year (60-120 total over 4 years) to minimize fee drag.
name = "6h_camarilla_pivot_1d_volume_trend_v1"
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
    
    # Get 1-day data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: based on previous day's OHLC
    # R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.1/2, R1 = C + (H-L)*0.5/2
    # S1 = C - (H-L)*0.5/2, S2 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    # where C = (H+L+C)/3 (typical price)
    
    # We need previous day's data to calculate today's levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Typical price of previous day
    prev_pivot = (prev_high + prev_low + prev_close) / 3.0
    prev_range = prev_high - prev_low
    
    # Camarilla levels for current day (based on previous day)
    R4 = prev_pivot + prev_range * 1.5 / 2.0
    R3 = prev_pivot + prev_range * 1.25 / 2.0
    S3 = prev_pivot - prev_range * 1.25 / 2.0
    S4 = prev_pivot - prev_range * 1.5 / 2.0
    
    # Align to 6h timeframe (these levels are constant throughout the day)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get 12h EMA for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_12h_6h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(R4_6h[i]) or np.isnan(R3_6h[i]) or 
            np.isnan(S3_6h[i]) or np.isnan(S4_6h[i]) or
            np.isnan(ema_12h_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or trend turns bearish
            if close[i] < S3_6h[i] or close[i] < ema_12h_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or trend turns bullish
            if close[i] > R3_6h[i] or close[i] > ema_12h_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Fade at R3/S3 in ranging markets, breakout at R4/S4 with volume
            # Only trade if volume confirms
            
            # Long conditions:
            # 1. Breakout above R4 with volume (trend continuation)
            # 2. Pullback to S3 with volume (mean reversion in uptrend)
            long_breakout = close[i] > R4_6h[i] and vol_ok and close[i] > ema_12h_6h[i]
            long_pullback = close[i] > S3_6h[i] and close[i] < R3_6h[i] and vol_ok and close[i] > ema_12h_6h[i]
            
            # Short conditions:
            # 1. Breakdown below S4 with volume
            # 2. Pullback to R3 with volume (mean reversion in downtrend)
            short_breakdown = close[i] < S4_6h[i] and vol_ok and close[i] < ema_12h_6h[i]
            short_pullback = close[i] < R3_6h[i] and close[i] > S3_6h[i] and vol_ok and close[i] < ema_12h_6h[i]
            
            if long_breakout or long_pullback:
                position = 1
                signals[i] = 0.25
            elif short_breakdown or short_pullback:
                position = -1
                signals[i] = -0.25
    
    return signals