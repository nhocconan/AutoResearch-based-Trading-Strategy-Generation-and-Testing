#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Donchian breakout with 1-day trend filter and volume confirmation
# Long when price breaks above weekly Donchian high with price > 1-day EMA50 and volume > 1.5x average
# Short when price breaks below weekly Donchian low with price < 1-day EMA50 and volume > 1.5x average
# Uses weekly price structure for institutional support/resistance, daily EMA for trend, volume for confirmation
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing
# Works in bull (breakouts) and bear (mean reversion via trend filter) by requiring alignment with daily trend

name = "6h_1wDonchian_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA50 on 1-day close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate weekly Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: 20-period rolling high
    donch_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: 20-period rolling low
    donch_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with uptrend and volume confirmation
            if close[i] > donch_high_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly Donchian low with downtrend and volume confirmation
            elif close[i] < donch_low_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (support break)
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (resistance break)
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals