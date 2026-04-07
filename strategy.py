#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market noise, capturing true trend direction.
# Combined with RSI momentum and Choppiness Index regime filter, it avoids
# whipsaws in sideways markets while catching trends in both bull and bear phases.
# Target: 10-25 trades/year (40-100 total over 4 years).

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
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
    
    # Get weekly data for trend filter and chop calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly KAMA for trend filter
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if np.sum(volatility[i-length+1:i+1]) > 0:
                er[i] = np.abs(close[i] - close[i-length]) / np.sum(volatility[i-length+1:i+1])
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(slow+1) - 2/(fast+1)) + 2/(fast+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_weekly = calculate_kama(close_weekly, 10, 2, 30)
    kama_weekly_prev = np.roll(kama_weekly, 1)  # Use previous week for no look-ahead
    kama_weekly_prev[0] = 0
    kama_trend = np.where(close_weekly > kama_weekly_prev, 1, -1)
    kama_trend_aligned = align_htf_to_ltf(prices, df_weekly, kama_trend)
    
    # Daily RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly Choppiness Index for regime filter
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(np.subtract(high, np.roll(close, 1)))
        tr3 = np.abs(np.subtract(low, np.roll(close, 1)))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=length, min_periods=1).sum().values
        
        hh = pd.Series(high).rolling(window=length, min_periods=1).max().values
        ll = pd.Series(low).rolling(window=length, min_periods=1).min().values
        
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if hh[i] - ll[i] != 0:
                chop[i] = 100 * np.log10(atr[i] / (hh[i] - ll[i])) / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    chop_weekly = calculate_chop(high_weekly, low_weekly, close_weekly, 14)
    chop_weekly_aligned = align_htf_to_ltf(prices, df_weekly, chop_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama_trend_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending
        chop_val = chop_weekly_aligned[i]
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 1:  # Long position
            # Exit: Trend turns bearish or RSI overbought in ranging market
            if kama_trend_aligned[i] == -1 or (is_ranging and rsi[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Trend turns bullish or RSI oversold in ranging market
            if kama_trend_aligned[i] == 1 or (is_ranging and rsi[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if is_trending:
                # Trend following: go with KAMA trend
                if kama_trend_aligned[i] == 1 and rsi[i] > 50:
                    position = 1
                    signals[i] = 0.25
                elif kama_trend_aligned[i] == -1 and rsi[i] < 50:
                    position = -1
                    signals[i] = -0.25
            elif is_ranging:
                # Mean reversion: fade extreme RSI
                if rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals