#!/usr/bin/env python3
# 4h_RSI_Divergence_4hTrend_Volume
# Hypothesis: Uses RSI divergence on 4h timeframe with 4h trend filter and volume confirmation.
# Long when bullish RSI divergence (price makes lower low, RSI makes higher low) with price above 4h EMA50 and volume above average.
# Short when bearish RSI divergence (price makes higher high, RSI makes lower high) with price below 4h EMA50 and volume above average.
# Exits when RSI returns to neutral zone (40-60) or trend reverses.
# Designed to work in both bull and bear markets by catching reversals at extremes.
# Targets 20-40 trades per year on 4h timeframe with position size 0.25.

name = "4h_RSI_Divergence_4hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) with proper min_periods
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Calculate EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema_50[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for RSI divergence (need at least 3 bars back)
        if i >= 3:
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_lower_low = low[i] < low[i-2] and low[i-1] > low[i]
            rsi_higher_low = rsi[i] > rsi[i-2] and rsi[i-1] > rsi[i]
            bullish_div = price_lower_low and rsi_higher_low
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            price_higher_high = high[i] > high[i-2] and high[i-1] < high[i]
            rsi_lower_high = rsi[i] < rsi[i-2] and rsi[i-1] < rsi[i]
            bearish_div = price_higher_high and rsi_lower_high
        else:
            bullish_div = False
            bearish_div = False
        
        if position == 0:
            # Long entry: bullish RSI divergence with price above EMA50 and volume confirmation
            if bullish_div and close[i] > ema_50[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish RSI divergence with price below EMA50 and volume confirmation
            elif bearish_div and close[i] < ema_50[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral zone (40-60) or trend reverses
            if rsi[i] >= 40 and rsi[i] <= 60 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral zone (40-60) or trend reverses
            if rsi[i] >= 40 and rsi[i] <= 60 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals