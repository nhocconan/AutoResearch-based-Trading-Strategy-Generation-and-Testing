# 1d_KAMA_RSI_Chop_Filter (revised) - Addressed too_few_trades by adding volume filter
# Hypothesis: KAMA adapts to trend, RSI identifies extremes, Chop identifies regimes.
# Volume filter ensures sufficient liquidity. Works in bull/bear by adapting to market regime.
# Target: 15-25 trades/year (60-100 total) with proper risk management.

#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter_v2"
timeframe = "1d"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - adapts to market noise
    def kama(close_prices, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        # Manual calculation to avoid pandas rolling apply
        er = np.zeros_like(close_prices, dtype=float)
        for i in range(length, len(close_prices)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama_values = np.zeros_like(close_prices)
        kama_values[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama_values[i] = kama_values[i-1] + sc[i] * (close_prices[i] - kama_values[i-1])
        return kama_values
    
    # RSI calculation
    def rsi(close_prices, length=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        avg_gain[length] = np.mean(gain[1:length+1])
        avg_loss[length] = np.mean(loss[1:length+1])
        
        for i in range(length+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values
    
    # Choppiness Index
    def chop(high_prices, low_prices, close_prices, length=14):
        # True Range
        tr1 = high_prices - low_prices
        tr2 = np.abs(high_prices - np.roll(close_prices, 1))
        tr3 = np.abs(low_prices - np.roll(close_prices, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_prices[0] - low_prices[0]  # First period
        
        # Sum of true ranges
        atr_sum = np.zeros_like(close_prices)
        for i in range(length, len(close_prices)):
            atr_sum[i] = np.sum(tr[i-length+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(high_prices)
        lowest_low = np.zeros_like(low_prices)
        for i in range(length-1, len(close_prices)):
            highest_high[i] = np.max(high_prices[i-length+1:i+1])
            lowest_low[i] = np.min(low_prices[i-length+1:i+1])
        
        # Choppiness calculation
        chop_values = np.zeros_like(close_prices)
        for i in range(length-1, len(close_prices)):
            if highest_high[i] != lowest_low[i]:
                chop_values[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(length)
            else:
                chop_values[i] = 50  # Neutral when no range
        return chop_values
    
    # Calculate indicators
    kama_vals = kama(close, length=10, fast=2, slow=30)
    rsi_vals = rsi(close, length=14)
    chop_vals = chop(high, low, close, length=14)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > 1.3 * vol_ma20
    
    # Get 1-week data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = np.zeros_like(close_1w)
    for i in range(34, len(close_1w)):
        ema34_1w[i] = np.mean(close_1w[i-34:i])  # Simple MA for stability
    trend_up_1w = close_1w > ema34_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN or invalid
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend) + RSI oversold + Chop indicates trend (not chop) + weekly uptrend + volume
            if (close[i] > kama_vals[i] and 
                rsi_vals[i] < 30 and 
                chop_vals[i] > 50 and  # Trending market
                trend_up_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) + RSI overbought + Chop indicates trend + weekly downtrend + volume
            elif (close[i] < kama_vals[i] and 
                  rsi_vals[i] > 70 and 
                  chop_vals[i] > 50 and  # Trending market
                  not trend_up_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below KAMA OR RSI overbought OR chop becomes too high (choppy)
            if (close[i] < kama_vals[i] or 
                rsi_vals[i] > 70 or 
                chop_vals[i] < 30):  # Too choppy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above KAMA OR RSI oversold OR chop becomes too high (choppy)
            if (close[i] > kama_vals[i] or 
                rsi_vals[i] < 30 or 
                chop_vals[i] < 30):  # Too choppy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals