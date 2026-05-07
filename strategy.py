#!/usr/bin/env python3
"""
4h_RSI_Divergence_Trend_Follow
Hypothesis: RSI divergence on 4h combined with higher timeframe trend (1d EMA50) and volume confirmation. 
Bullish divergence: price makes lower low, RSI makes higher low. Bearish divergence: price makes higher high, RSI makes lower high.
Long on bullish divergence in uptrend (price > 1d EMA50) with volume spike. Short on bearish divergence in downtrend (price < 1d EMA50) with volume spike.
Designed for 20-40 trades/year to minimize fee drag. Works in bull/bear via 1d trend filter and divergence signals that capture exhaustion before reversals.
"""

name = "4h_RSI_Divergence_Trend_Follow"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(close_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Check for RSI divergence (need at least 3 bars back)
        if i >= 3:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = (low[i] < low[i-2] and 
                       rsi[i] > rsi[i-2] and
                       low[i-1] < low[i-3] and 
                       rsi[i-1] > rsi[i-3])
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = (high[i] > high[i-2] and 
                       rsi[i] < rsi[i-2] and
                       high[i-1] > high[i-3] and 
                       rsi[i-1] < rsi[i-3])
        else:
            bull_div = False
            bear_div = False
        
        if position == 0:
            # Long: bullish divergence in uptrend with volume spike
            if bull_div and trend_up and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence in downtrend with volume spike
            elif bear_div and trend_down and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish divergence or trend turns down
            if bear_div or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish divergence or trend turns up
            if bull_div or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals