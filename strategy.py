#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_rsi_divergence_volume_v1
# RSI divergence on 12h timeframe with volume confirmation on 4h for entry timing.
# In bull markets: bullish divergence (price lower low, RSI higher low) + volume spike = long.
# In bear markets: bearish divergence (price higher high, RSI lower high) + volume spike = short.
# Uses RSI(14) on 12h to capture major trend exhaustion points, volume to confirm institutional participation.
# Target: 15-30 trades/year per symbol for low friction and high edge.
name = "4h_12h_rsi_divergence_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for RSI calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate RSI on 12h close
    rsi_period = 14
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if RSI not ready
        if np.isnan(rsi_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if volume confirmation fails
        if not vol_confirm[i]:
            # Hold current position if volume fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Need at least 3 bars of history for divergence check
        if i < 3:
            signals[i] = 0.0
            continue
        
        # Get current and past values for divergence detection
        rsi_now = rsi_12h_aligned[i]
        rsi_prev = rsi_12h_aligned[i-1]
        rsi_prev2 = rsi_12h_aligned[i-2]
        
        price_now = close[i]
        price_prev = close[i-1]
        price_prev2 = close[i-2]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = (price_now < price_prev and price_prev < price_prev2) and \
                      (rsi_now > rsi_prev and rsi_prev > rsi_prev2)
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = (price_now > price_prev and price_prev > price_prev2) and \
                      (rsi_now < rsi_prev and rsi_prev < rsi_prev2)
        
        # Entry signals with volume confirmation
        if bullish_div and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_div and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit on opposite divergence
        elif bearish_div and position == 1:
            position = 0
            signals[i] = 0.0
        elif bullish_div and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals