#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA trend filter and volume spike.
# Williams %R measures overbought/oversold levels: -20 to 0 = overbought, -80 to -100 = oversold.
# Strategy: In trending markets (12h price > EMA50), buy oversold pullbacks (%R < -80) and sell overbought rallies (%R > -20).
# In ranging markets (12h price near EMA50), fade extremes (%R < -85 for long, %R > -15 for short).
# Volume spike confirms institutional participation. Designed for ~15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    diff = highest_high - lowest_low
    diff = np.where(diff == 0, 1e-10, diff)
    willr = -100 * (highest_high - close) / diff
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime based on 12h price vs EMA50
        price_vs_ema = close_12h[i // 2] if i >= 2 else close_12h[0]  # Approximate 12h price from 6h data
        # Better: use aligned 12h close price
        # Since we don't have 12h close aligned, we'll use the EMA as trend proxy
        is_uptrend = close[i] > ema50_12h_aligned[i]
        is_downtrend = close[i] < ema50_12h_aligned[i]
        is_ranging = abs(close[i] - ema50_12h_aligned[i]) < (ema50_12h_aligned[i] * 0.01)  # Within 1% of EMA
        
        if is_uptrend and volume_filter[i]:
            # Buy oversold pullbacks in uptrend
            if willr[i] < -80:
                signals[i] = 0.25
                position = 1
            # Exit if overbought
            elif willr[i] > -20 and position == 1:
                signals[i] = 0.0
                position = 0
        elif is_downtrend and volume_filter[i]:
            # Sell overbought rallies in downtrend
            if willr[i] > -20:
                signals[i] = -0.25
                position = -1
            # Exit if oversold
            elif willr[i] < -80 and position == -1:
                signals[i] = 0.0
                position = 0
        elif is_ranging:
            # Fade extremes in ranging market
            if willr[i] < -85 and volume_filter[i]:  # Deep oversold
                signals[i] = 0.25
                position = 1
            elif willr[i] > -15 and volume_filter[i]:  # Deep overbought
                signals[i] = -0.25
                position = -1
            # Exit on mean reversion
            elif -50 < willr[i] < -30 and position == 1:
                signals[i] = 0.0
                position = 0
            elif -30 < willr[i] < -50 and position == -1:
                signals[i] = 0.0
                position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_12hEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0