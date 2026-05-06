#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Elder Ray (Bull/Bear Power) with trend filter and volume confirmation
# Long when Bull Power > 0, price > 200-bar EMA, and volume > 1.5x average
# Short when Bear Power < 0, price < 200-bar EMA, and volume > 1.5x average
# Uses Elder Ray to measure bull/bear strength relative to EMA13, EMA200 for trend filter, volume for confirmation
# Target: 15-30 trades per year (60-120 over 4 years) with 0.25 position sizing

name = "6h_1dElderRay_EMA200_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA200 on 6h close (needs 200 bars)
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate EMA13 on 1-day high/low for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # EMA13 of daily high and low
    ema13_high = pd.Series(df_1d['high'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_low = pd.Series(df_1d['low'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13(High), Bear Power = Low - EMA13(Low)
    bull_power = df_1d['high'].values - ema13_high
    bear_power = df_1d['low'].values - ema13_low
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema200[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish bias), uptrend, volume confirmation
            if bull_power_aligned[i] > 0 and close[i] > ema200[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish bias), downtrend, volume confirmation
            elif bear_power_aligned[i] < 0 and close[i] < ema200[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power turns negative (bearish bias)
            if bear_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power turns positive (bullish bias)
            if bull_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals