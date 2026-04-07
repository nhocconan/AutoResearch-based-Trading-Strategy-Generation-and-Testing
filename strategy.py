#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Daily KAMA with Weekly Trend Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals in both bull and bear markets.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation ensures institutional participation. Target: 15-25 trades/year.

name = "1d_kama_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend direction
    weekly_close = df_weekly['close'].values
    weekly_close_series = pd.Series(weekly_close)
    # Efficiency ratio
    change = abs(weekly_close_series.diff(10))
    volatility = weekly_close_series.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama = np.zeros_like(weekly_close)
    kama[0] = weekly_close[0]
    for i in range(1, len(weekly_close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (weekly_close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    # Trend: 1 if close > KAMA, -1 if close < KAMA
    weekly_trend = np.where(weekly_close > kama, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend)
    
    # Daily KAMA for entry signal
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = close_series.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama_daily = np.zeros(n)
    kama_daily[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]):
            kama_daily[i] = kama_daily[i-1] + sc.iloc[i] * (close[i] - kama_daily[i-1])
        else:
            kama_daily[i] = kama_daily[i-1]
    
    # Volume filter: volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama_daily[i]) or np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or weekly trend turns bearish or volume drops
            if (close[i] < kama_daily[i] or weekly_trend_aligned[i] == -1 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or weekly trend turns bullish or volume drops
            if (close[i] > kama_daily[i] or weekly_trend_aligned[i] == 1 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price crosses above KAMA with weekly bullish trend and volume
            if (close[i] > kama_daily[i] and weekly_trend_aligned[i] == 1 and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price crosses below KAMA with weekly bearish trend and volume
            elif (close[i] < kama_daily[i] and weekly_trend_aligned[i] == -1 and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals