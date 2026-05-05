#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R4 AND volume > 2.0x 20-period average AND 1w EMA50 uptrend
# Short when price breaks below Camarilla S4 AND volume > 2.0x 20-period average AND 1w EMA50 downtrend
# Exit when price crosses Camarilla Pivot point OR 1w trend reverses
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-25 trades/year per symbol.
# Camarilla R4/S4 are stronger breakout levels than R3/S3, reducing false breakouts.
# 1w EMA50 provides major trend filter to avoid counter-trend whipsaws in both bull and bear markets.
# Volume spike confirms institutional participation. Strategy avoids overtrading by using strict conditions.

name = "4h_Camarilla_R4S4_VolumeSpike_1wEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h data (based on previous day's OHLC)
    # We'll use the previous 4h bar's OHLC to calculate today's levels
    # For simplicity, we use rolling window of 6 bars (1 day = 6*4h) to get daily OHLC
    if len(df_4h) < 6:
        return np.zeros(n)
    
    # Calculate daily OHLC from 4h data (6 bars = 1 day)
    daily_open = df_4h['open'].rolling(window=6, min_periods=6).first().values
    daily_high = df_4h['high'].rolling(window=6, min_periods=6).max().values
    daily_low = df_4h['low'].rolling(window=6, min_periods=6).min().values
    daily_close = df_4h['close'].rolling(window=6, min_periods=6).last().values
    
    # Camarilla levels: based on previous day's range
    # R4/S4 are stronger levels (1.1/2 multiplier vs 1.1/4 for R3/S3)
    R4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    S4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    Pivot = (daily_high + daily_low + daily_close) / 3
    
    # Align Camarilla levels to prices timeframe
    R4_aligned = align_htf_to_ltf(prices, df_4h, R4)
    S4_aligned = align_htf_to_ltf(prices, df_4h, S4)
    Pivot_aligned = align_htf_to_ltf(prices, df_4h, Pivot)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 4h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)  # No volume confirmation if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(Pivot_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND volume spike AND 1w EMA50 uptrend
            if (close[i] > R4_aligned[i] and 
                volume_filter[i] and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND volume spike AND 1w EMA50 downtrend
            elif (close[i] < S4_aligned[i] and 
                  volume_filter[i] and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla Pivot OR 1w trend changes to downtrend
            if (close[i] < Pivot_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla Pivot OR 1w trend changes to uptrend
            if (close[i] > Pivot_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals