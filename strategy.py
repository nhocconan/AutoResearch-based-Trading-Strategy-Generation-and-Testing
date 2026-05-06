#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Camarilla pivot (R3/S3) breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above daily Camarilla R3 with EMA50 trend alignment and volume > 1.5x average
# Short when price breaks below daily Camarilla S3 with EMA50 trend alignment and volume > 1.5x average
# Daily Camarilla provides strong intraday support/resistance, EMA50 filters counter-trend moves,
# Volume confirms breakout strength. Designed for low trade frequency and high win rate.
# Target: 20-50 trades per year (80-200 over 4 years) with 0.25 position sizing.

name = "4h_Camarilla_R3S3_EMA50_Volume_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 2.0)  # R3 = pivot + 1.1*(H-L)/2
    s3 = pivot - (range_1d * 1.1 / 2.0)  # S3 = pivot - 1.1*(H-L)/2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above daily Camarilla R3 with EMA50 uptrend and volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema_50[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below daily Camarilla S3 with EMA50 downtrend and volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema_50[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily Camarilla S3 (mean reversion) or EMA50 turns down
            if close[i] < s3_aligned[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily Camarilla R3 (mean reversion) or EMA50 turns up
            if close[i] > r3_aligned[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals