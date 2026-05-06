#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily pivot levels with volume confirmation and trend filter
# Uses daily pivot points (PP, R1, S1, R2, S2) for institutional support/resistance
# Price breaking above R2 or below S2 with volume > 1.5x average indicates breakout
# Price rejecting at R1 or S1 with volume confirmation indicates mean reversion
# Trend filter: 4h EMA(50) to avoid counter-trend trades in strong trends
# Target: 80-150 total trades over 4 years (20-38/year) with 0.25 position sizing

name = "4h_PivotPoints_R2S2_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Resistance and Support levels
    r2 = pivot + range_
    r1 = pivot + (prev_high - prev_low) / 2
    s1 = pivot - (prev_high - prev_low) / 2
    s2 = pivot - range_
    
    # Align daily levels to 4h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Trend filter: 4h EMA(50)
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_filter = close > ema_50  # bullish bias
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r2_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(trend_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume confirmation and bullish trend
            if close[i] > r2_aligned[i] and volume_filter[i] and trend_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S2 with volume confirmation and bearish trend
            elif close[i] < s2_aligned[i] and volume_filter[i] and not trend_filter[i]:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects S1 with volume confirmation in bullish trend
            elif close[i] < s1_aligned[i] and close[i] > s1_aligned[i] * 0.995 and volume_filter[i] and trend_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects R1 with volume confirmation in bearish trend
            elif close[i] > r1_aligned[i] and close[i] < r1_aligned[i] * 1.005 and volume_filter[i] and not trend_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (failed support) or reaches R2 (take profit)
            if close[i] < s1_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (failed resistance) or reaches S2 (take profit)
            if close[i] > r1_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals