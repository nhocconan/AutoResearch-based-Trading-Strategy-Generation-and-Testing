# 6h_Linear_Trend_Retracement_v1
# Hypothesis: In 6h timeframe, price often retraces to linear trend channels (upper/lower) before continuing trend.
# Uses linear regression slope to identify trend direction and dynamic support/resistance levels.
# Long when price touches lower channel in uptrend; short when price touches upper channel in downtrend.
# Works in both bull and bear markets by following trend direction from higher timeframe (12h).
# Volume confirmation ensures institutional participation.
# Target: 50-150 trades over 4 years with disciplined entries.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Linear_Trend_Retracement_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend direction (more stable than 1d for 6h entries)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h linear regression slope for trend (20-period)
    close_12h = df_12h['close'].values
    slope_12h = np.full(len(close_12h), np.nan)
    for i in range(19, len(close_12h)):
        y = close_12h[i-19:i+1]
        x = np.arange(20)
        if np.all(~np.isnan(y)):
            slope = np.polyfit(x, y, 1)[0]
            slope_12h[i] = slope
    slope_12h_aligned = align_htf_to_ltf(prices, df_12h, slope_12h)
    
    # 6h linear regression channel (20-period) for support/resistance
    slope_6h = np.full(n, np.nan)
    intercept_6h = np.full(n, np.nan)
    for i in range(19, n):
        y = close[i-19:i+1]
        x = np.arange(20)
        if np.all(~np.isnan(y)):
            slope, intercept = np.polyfit(x, y, 1)
            slope_6h[i] = slope
            intercept_6h[i] = intercept
    
    # Upper and lower channel (1 ATR width from regression line)
    atr_raw = np.abs(high - low)
    atr = pd.Series(atr_raw).rolling(window=14, min_periods=14).mean().values
    channel_width = atr  # 1 ATR width
    
    upper_channel = slope_6h * np.arange(n) + intercept_6h + channel_width
    lower_channel = slope_6h * np.arange(n) + intercept_6h - channel_width
    
    # Volume confirmation: volume > 1.2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.2 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(slope_12h_aligned[i]) or np.isnan(slope_6h[i]) or 
            np.isnan(intercept_6h[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: Uptrend (12h slope > 0) + price touches lower channel + volume
            if (slope_12h_aligned[i] > 0 and 
                price <= lower_channel[i] * 1.001 and  # Allow small tolerance
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend (12h slope < 0) + price touches upper channel + volume
            elif (slope_12h_aligned[i] < 0 and 
                  price >= upper_channel[i] * 0.999 and  # Allow small tolerance
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reaches middle of channel or trend changes
            middle_channel = (upper_channel[i] + lower_channel[i]) / 2
            if price >= middle_channel or slope_12h_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reaches middle of channel or trend changes
            middle_channel = (upper_channel[i] + lower_channel[i]) / 2
            if price <= middle_channel or slope_12h_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals