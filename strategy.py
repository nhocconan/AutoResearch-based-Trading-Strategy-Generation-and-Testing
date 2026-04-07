#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h KAMA Trend + 1d Volume Filter + 1w Regime
# Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
# in ranging markets it stays flat. We use 1d volume to confirm institutional participation
# and 1week trend filter (price > 50-period SMA) to avoid counter-trend trades.
# Works in both bull and bear markets by only trading with the higher timeframe trend.
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag.
name = "4h_kama_1d_volume_1w_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA parameters
    fast_sc = 2/(2+1)   # EMA constant for fastest EMA
    slow_sc = 2/(30+1)  # EMA constant for slowest EMA
    
    # Calculate Efficiency Ratio and KAMA
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1-day volume filter: current volume > 1.5x 20-period average
    daily_volume = df_1d['volume'].values
    vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean()
    vol_filter_1d = daily_volume > (vol_ma * 1.5)
    vol_filter_1d_4h = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    # 1-week trend filter: price > 50-period SMA
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_4h = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(vol_filter_1d_4h[i]) or 
            np.isnan(weekly_sma50_4h[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime from 1w data
        uptrend = close[i] > weekly_sma50_4h[i]  # Uptrend when price > weekly SMA50
        downtrend = close[i] < weekly_sma50_4h[i]  # Downtrend when price < weekly SMA50
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or trend turns down
            if close[i] < kama[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or trend turns up
            if close[i] > kama[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter_1d_4h[i]:
                # Go long if price > KAMA and in uptrend
                if close[i] > kama[i] and uptrend:
                    position = 1
                    signals[i] = 0.25
                # Go short if price < KAMA and in downtrend
                elif close[i] < kama[i] and downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals