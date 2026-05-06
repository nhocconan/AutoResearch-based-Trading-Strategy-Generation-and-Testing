#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Donchian(20) breakout with 1w EMA20 trend filter and volume confirmation
# Long when price breaks above 20-week high with price > 20-week EMA and volume > 1.8x average
# Short when price breaks below 20-week low with price < 20-week EMA and volume > 1.8x average
# Uses 1w Donchian channels for institutional support/resistance, EMA20 for trend filter, volume for confirmation
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing

name = "1d_1wDonchian20_EMA20_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA20 on 1d close (needs 20 bars)
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate 1-week Donchian(20) channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-week high and low
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Volume confirmation: >1.8x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.8 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after EMA20 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 20-week high with uptrend and volume confirmation
            if close[i] > high_20w_aligned[i] and close[i] > ema20[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 20-week low with downtrend and volume confirmation
            elif close[i] < low_20w_aligned[i] and close[i] < ema20[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-week low (support break)
            if close[i] < low_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-week high (resistance break)
            if close[i] > high_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals