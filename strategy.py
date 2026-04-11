#!/usr/bin/env python3
# 12h_1d_rsi_divergence_volume_v1
# Strategy: 12-hour RSI divergence with volume confirmation and 1-day trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: RSI divergence captures momentum exhaustion with high accuracy.
# Bullish when bullish RSI divergence forms (price makes lower low, RSI makes higher low) with volume confirmation and price above 1-day EMA50.
# Bearish when bearish RSI divergence forms (price makes higher high, RSI makes lower high) with volume confirmation and price below 1-day EMA50.
# Works in bull markets by catching pullbacks and in bear markets by catching bounces.
# Uses tight entry conditions to limit trades (~15-30/year) and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_rsi_divergence_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).values
    
    # 12h Volume confirmation: current volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_avg_20)
    
    # RSI divergence detection
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(14, n):
        # Look for bullish divergence: price makes lower low, RSI makes higher low
        if low[i] < low[i-1] and rsi[i] > rsi[i-1]:
            # Check if this is a significant low point
            if i >= 28:  # Need at least 2 periods to compare
                # Find recent low in price and RSI
                lookback = min(20, i//2)
                price_low_idx = i - np.argmin(l[i-lookback:i+1]) if lookback > 0 else i
                rsi_low_idx = i - np.argmin(rsi[i-lookback:i+1]) if lookback > 0 else i
                
                # Bullish divergence: price lower low, RSI higher low
                if (price_low_idx == i and rsi_low_idx < i and 
                    low[i] < low[price_low_idx] and rsi[i] > rsi[rsi_low_idx]):
                    bullish_div[i] = True
        
        # Look for bearish divergence: price makes higher high, RSI makes lower high
        if high[i] > high[i-1] and rsi[i] < rsi[i-1]:
            # Check if this is a significant high point
            if i >= 28:
                lookback = min(20, i//2)
                price_high_idx = i - np.argmax(h[i-lookback:i+1]) if lookback > 0 else i
                rsi_high_idx = i - np.argmax(rsi[i-lookback:i+1]) if lookback > 0 else i
                
                # Bearish divergence: price higher high, RSI lower high
                if (price_high_idx == i and rsi_high_idx < i and 
                    high[i] > high[price_high_idx] and rsi[i] < rsi[rsi_high_idx]):
                    bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after RSI warmup
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry logic: RSI divergence + volume + trend alignment
        if bullish_div[i] and vol_confirm[i] and close[i] > ema_50_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_div[i] and vol_confirm[i] and close[i] < ema_50_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite divergence with volume confirmation
        elif position == 1 and bearish_div[i] and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish_div[i] and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals