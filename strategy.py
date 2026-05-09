#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h RSI(14) with 1d EMA(50) trend filter and volume confirmation
# RSI captures momentum extremes while avoiding overtrading through high timeframe (12h).
# EMA(50) on daily filters trend direction to avoid counter-trend trades.
# Volume > 1.3x 20-period average confirms institutional participation.
# Designed for 12-37 trades/year (50-150 total over 4 years) with controlled risk.
name = "12h_RSI14_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 12h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and RSI calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        trend_up = close[i] > ema_50_12h[i]
        trend_down = close[i] < ema_50_12h[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) + uptrend + volume confirmation
            if rsi_val < 30 and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + downtrend + volume confirmation
            elif rsi_val > 70 and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (momentum fading) or trend reversal
            if rsi_val > 50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 (momentum fading) or trend reversal
            if rsi_val < 50 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals