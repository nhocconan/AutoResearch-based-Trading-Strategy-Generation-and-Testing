#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily KAMA with weekly trend filter and volume confirmation
# Hypothesis: KAMA adapts to market conditions, reducing whipsaw in sideways markets.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation adds institutional participation validation.
# Target: 10-25 trades/year to minimize fee drag on daily timeframe.
name = "daily_kama_weekly_trend_volume_v1"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend direction
    close_1w = df_1w['close'].values
    # Calculate Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_1w, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)  # 1-period volatility sum
    # Handle first 10 values
    change = np.concatenate([[np.nan]*10, change])
    volatility = np.concatenate([[np.nan]*10, volatility[10:]])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if not np.isnan(sc[i]):
            kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
        else:
            kama_1w[i] = kama_1w[i-1]
    kama_1w = np.concatenate([[np.nan]*9, kama_1w[10:]])  # align with original indexing
    
    # Trend: price above/below weekly KAMA
    trend_1w = close_1w > kama_1w
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w.astype(float))
    
    # Calculate daily KAMA for entry signal
    change_d = np.abs(np.diff(close, n=10))
    volatility_d = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    change_d = np.concatenate([[np.nan]*10, change_d])
    volatility_d = np.concatenate([[np.nan]*10, volatility_d[10:]])
    er_d = np.where(volatility_d != 0, change_d / volatility_d, 0)
    sc_d = (er_d * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_d = np.full_like(close, np.nan)
    kama_d[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc_d[i]):
            kama_d[i] = kama_d[i-1] + sc_d[i] * (close[i] - kama_d[i-1])
        else:
            kama_d[i] = kama_d[i-1]
    
    # Calculate daily 20-period volume moving average
    vol_ma_d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(kama_d[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(vol_ma_d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 20-day average
        vol_confirm = volume[i] > vol_ma_d[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below daily KAMA OR trend changes
            if close[i] < kama_d[i] or trend_1w_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above daily KAMA OR trend changes
            if close[i] > kama_d[i] or trend_1w_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price above daily KAMA, uptrend on weekly, volume confirmation
            if (close[i] > kama_d[i] and trend_1w_aligned[i] > 0.5 and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Enter short: price below daily KAMA, downtrend on weekly, volume confirmation
            elif (close[i] < kama_d[i] and trend_1w_aligned[i] < 0.5 and vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals