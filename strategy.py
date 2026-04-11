#!/usr/bin/env python3
# 4h_1d_rsi_divergence_volume_v1
# Strategy: 4h RSI divergence with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: RSI divergences at key turning points combined with 1d trend alignment and volume spikes capture reversals with high accuracy. Works in both bull and bear markets by identifying exhaustion points. Targets 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rsi_divergence_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate RSI(14) on 4h data
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # Initialize first average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            # Wilder smoothing
            for i in range(period, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Find pivot points for divergence detection
    def find_pivots(arr, left=3, right=3):
        """Find pivot highs and lows"""
        n = len(arr)
        highs = np.zeros(n, dtype=bool)
        lows = np.zeros(n, dtype=bool)
        
        for i in range(left, n - right):
            # Check for pivot high
            if all(arr[i] >= arr[i-left:i]) and all(arr[i] >= arr[i+1:i+right+1]):
                highs[i] = True
            # Check for pivot low
            if all(arr[i] <= arr[i-left:i]) and all(arr[i] <= arr[i+1:i+right+1]):
                lows[i] = True
        return highs, lows
    
    # Find price and RSI pivots
    price_highs, price_lows = find_pivots(close, 3, 3)
    rsi_highs, rsi_lows = find_pivots(rsi, 3, 3)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if price_lows[i] and rsi_lows[i]:
            # Look back for previous pivot low
            for j in range(i-1, max(0, i-20), -1):
                if price_lows[j] and rsi_lows[j]:
                    if close[i] < close[j] and rsi[i] > rsi[j]:
                        bullish_div = True
                    break
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if price_highs[i] and rsi_highs[i]:
            # Look back for previous pivot high
            for j in range(i-1, max(0, i-20), -1):
                if price_highs[j] and rsi_highs[j]:
                    if close[i] > close[j] and rsi[i] < rsi[j]:
                        bearish_div = True
                    break
        
        # 1d EMA trend filter
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Bullish divergence AND bearish trend (reversal) AND volume confirmation
        if bullish_div and trend_bearish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Bearish divergence AND bullish trend (reversal) AND volume confirmation
        elif bearish_div and trend_bullish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite divergence or trend reversal
        elif position == 1 and (bearish_div or trend_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_div or trend_bearish):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals