#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Donchian breakouts with volume confirmation and trend filter
# Daily Donchian breakouts capture strong trends. Volume > 1.8x 30-period average confirms momentum.
# 12h EMA50 trend filter ensures we only trade in direction of intermediate trend.
# Works in bull/bear markets: breakouts capture trends, EMA filter prevents counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1dDonchian20_EMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's Donchian (to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # 20-period Donchian channels
    upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Volume confirmation: >1.8x 30-period average (moderate threshold)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.8 * vol_ma_30)
    
    # Trend filter: 12h EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with volume and trend filter
            if close[i] > upper_aligned[i] and volume_filter[i] and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian with volume and trend filter
            elif close[i] < lower_aligned[i] and volume_filter[i] and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (trend reversal)
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (trend reversal)
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals