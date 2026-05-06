#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI divergence + 1d EMA50 trend + volume confirmation
# Uses RSI divergence (bullish/bearish) for early reversal signals
# 1d EMA50 filters for trend alignment to avoid counter-trend trades
# Volume spike (>2x 20-bar average) confirms momentum behind the move
# Discrete sizing 0.25 targets ~100 total trades over 4 years (25/year)
# Works in bull/bear: divergences catch reversals, trend filter avoids false signals, volume ensures participation

name = "4h_RSIDivergence_1dEMA50_VolumeConfirm_v1"
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
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume filter (>2x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe (primary)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or np.isnan(rsi[i-2]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = (low[i] < low[i-1] and low[i-1] < low[i-2] and 
                          rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2])
            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = (high[i] > high[i-1] and high[i-1] > high[i-2] and 
                          rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2])
            
            # Long: bullish divergence + uptrend (price > EMA50) + volume spike
            if bullish_div and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + downtrend (price < EMA50) + volume spike
            elif bearish_div and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI overbought (>70) or trend change (price < EMA50)
            if rsi[i] > 70 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold (<30) or trend change (price > EMA50)
            if rsi[i] < 30 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals