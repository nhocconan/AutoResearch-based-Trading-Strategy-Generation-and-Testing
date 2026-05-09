#!/usr/bin/env python3
# Hypothesis: 4h RSI divergence with 12h EMA trend filter and volume confirmation
# Long when price makes lower low but RSI makes higher low (bullish divergence) with 12h EMA uptrend and volume > 1.5x average
# Short when price makes higher high but RSI makes lower high (bearish divergence) with 12h EMA downtrend and volume > 1.5x average
# Exit when RSI crosses 50 or opposite divergence occurs
# Uses RSI divergence for early reversal signals, EMA for trend filter, volume for confirmation
# Designed to work in both trending and ranging markets with controlled frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_RSI_Divergence_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    # Detect RSI divergence
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    # Look for price lows and RSI lows for bullish divergence
    for i in range(2, n-2):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (low[i] < low[i-1] and low[i] < low[i+1] and 
            rsi[i] > rsi[i-1] and rsi[i] > rsi[i+1]):
            # Check if this is a meaningful low (look back 5 periods)
            if i >= 5:
                price_low_5b = np.min(low[i-5:i])
                rsi_low_5b = np.min(rsi[i-5:i])
                if low[i] == price_low_5b and rsi[i] == rsi_low_5b:
                    bullish_div[i] = True
    
    # Look for price highs and RSI highs for bearish divergence
    for i in range(2, n-2):
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (high[i] > high[i-1] and high[i] > high[i+1] and 
            rsi[i] < rsi[i-1] and rsi[i] < rsi[i+1]):
            # Check if this is a meaningful high (look back 5 periods)
            if i >= 5:
                price_high_5b = np.max(high[i-5:i])
                rsi_high_5b = np.max(rsi[i-5:i])
                if high[i] == price_high_5b and rsi[i] == rsi_high_5b:
                    bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for RSI and EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish RSI divergence, 12h EMA uptrend, volume confirmation
            if (bullish_div[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish RSI divergence, 12h EMA downtrend, volume confirmation
            elif (bearish_div[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses below 50 or bearish divergence
            if (rsi[i] < 50) or bearish_div[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses above 50 or bullish divergence
            if (rsi[i] > 50) or bullish_div[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals