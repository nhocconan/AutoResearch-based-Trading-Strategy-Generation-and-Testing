#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(14) divergence with 1d EMA50 trend filter and volume confirmation
# Bullish divergence: price makes lower low while RSI makes higher low -> long
# Bearish divergence: price makes higher high while RSI makes lower high -> short
# Trend filter: only take longs when price > 1d EMA50, shorts when price < 1d EMA50
# Volume confirmation: volume > 1.5x 20-period average to ensure participation
# Discrete sizing 0.25 to limit risk; target 60-120 total trades over 4 years (15-30/year)
# Works in bull markets via momentum continuation, in bear via mean reversion at extremes

name = "4h_RSIDivergence_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume confirmation (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (i >= 2 and low[i] < low[i-1] and low[i-1] < low[i-2] and 
                rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2] and
                close[i] > ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (i >= 2 and high[i] > high[i-1] and high[i-1] > high[i-2] and 
                  rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2] and
                  close[i] < ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 70 (overbought) or price < EMA50
            if rsi[i] > 70 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 30 (oversold) or price > EMA50
            if rsi[i] < 30 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals