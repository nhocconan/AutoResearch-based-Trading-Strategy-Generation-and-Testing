#!/usr/bin/env python3
# 6H_ELLIOTT_WAVE_COUNT_1D_SUPERTREND
# Hypothesis: Elliot Wave theory suggests price moves in 5-wave impulse patterns followed by 3-wave corrections.
# In strong trends, wave 3 is typically the strongest and most extended.
# We use 1d Supertrend to identify the primary trend direction, then on 6h chart we look for
# momentum acceleration (wave 3) using RSI divergence and price action.
# Entry: In uptrend (1d Supertrend bullish), look for bullish RSI divergence + price above 6h EMA20.
# Entry: In downtrend (1d Supertrend bearish), look for bearish RSI divergence + price below 6h EMA20.
# Exit: Opposite Supertrend signal or RSI exits extreme zone.
# This captures strong momentum moves while avoiding counter-trend trades.
# Target: 15-30 trades/year on 6h timeframe.

name = "6H_ELLIOTT_WAVE_COUNT_1D_SUPERTREND"
timeframe = "6h"
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
    
    # 1D data for Supertrend and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Supertrend (ATR=10, multiplier=3.0)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic upper and lower bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    start_idx = 10
    if len(close_1d) > start_idx:
        supertrend[start_idx] = upper_band[start_idx]
        direction[start_idx] = 1  # Start in uptrend
        
        for i in range(start_idx + 1, len(close_1d)):
            if close_1d[i] > supertrend[i-1]:
                # Uptrend
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                # Downtrend
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
    
    # RSI for divergence detection (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
            
            # Wilder's smoothing
            for i in range(period+1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        # Set first period values to NaN
        rsi[:period] = np.nan
        return rsi
    
    rsi_1d = calculate_rsi(close_1d)
    
    # 6H indicators
    # EMA20 for dynamic support/resistance
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI on 6h for entry timing
    rsi_6h = calculate_rsi(close)
    
    # Align 1D data to 6H
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_6h[i]) or np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend (1d Supertrend bullish) + bullish RSI divergence + price above 6h EMA20
            # Bullish divergence: price making lower low, RSI making higher low
            bullish_div = False
            if i >= 20:  # Need sufficient lookback
                # Check for price lower low and RSI higher low over last 10 periods
                if (low[i] < low[i-5] and low[i-5] < low[i-10] and 
                    rsi_6h[i] > rsi_6h[i-5] and rsi_6h[i-5] > rsi_6h[i-10]):
                    bullish_div = True
            
            if (direction_aligned[i] == 1 and  # 1d Uptrend
                bullish_div and 
                close[i] > ema20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (1d Supertrend bearish) + bearish RSI divergence + price below 6h EMA20
            # Bearish divergence: price making higher high, RSI making lower high
            bearish_div = False
            if i >= 20:
                if (high[i] > high[i-5] and high[i-5] > high[i-10] and 
                    rsi_6h[i] < rsi_6h[i-5] and rsi_6h[i-5] < rsi_6h[i-10]):
                    bearish_div = True
            
            if (direction_aligned[i] == -1 and  # 1d Downtrend
                bearish_div and 
                close[i] < ema20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or RSI overbought
            if (direction_aligned[i] == -1 or  # 1d trend turned bearish
                rsi_6h[i] > 70):  # RSI overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or RSI oversold
            if (direction_aligned[i] == 1 or  # 1d trend turned bullish
                rsi_6h[i] < 30):  # RSI oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals