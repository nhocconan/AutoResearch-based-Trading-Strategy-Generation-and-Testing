#!/usr/bin/env python3
# 1d_KAMA_Trend_With_RSI_Filter_v2
# Hypothesis: On daily timeframe, use KAMA (Kaufman Adaptive Moving Average) to determine trend direction.
# Enter long when price crosses above KAMA and RSI(14) > 50, short when price crosses below KAMA and RSI(14) < 50.
# Use weekly timeframe for trend filter: only take longs when price > weekly EMA(50), shorts when price < weekly EMA(50).
# Add volume confirmation: require current volume > 1.5x 20-day average volume.
# Designed for 10-25 trades/year on 1d timeframe to avoid fee drag while capturing major trends.

name = "1d_KAMA_Trend_With_RSI_Filter_v2"
timeframe = "1d"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on close prices
    # Parameters: ER fast = 2, slow = 30, lookback = 10 for efficiency ratio
    def calculate_kama(close_prices, fast=2, slow=30, lookback=10):
        kama = np.full_like(close_prices, np.nan, dtype=np.float64)
        if len(close_prices) < lookback + 1:
            return kama
        
        # Efficiency Ratio
        change = np.abs(close_prices[lookback:] - close_prices[:-lookback])
        volatility = np.sum(np.abs(np.diff(close_prices[lookback-1:])), axis=0) if lookback > 1 else np.abs(np.diff(close_prices[lookback-1:]))
        # Handle volatility calculation properly
        volatility = np.full_like(close_prices, np.nan)
        for i in range(lookback, len(close_prices)):
            volatility[i] = np.sum(np.abs(np.diff(close_prices[i-lookback+1:i+1])))
        
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        
        # Initialize KAMA
        kama[lookback] = close_prices[lookback]
        for i in range(lookback + 1, len(close_prices)):
            if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, fast=2, slow=30, lookback=10)
    
    # Calculate RSI(14)
    def calculate_rsi(close_prices, period=14):
        rsi = np.full_like(close_prices, np.nan, dtype=np.float64)
        if len(close_prices) < period + 1:
            return rsi
        
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        # Initial average
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(50)
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50
    
    # Align weekly EMA to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: daily volume / 20-day average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price crosses above KAMA AND RSI > 50 AND weekly uptrend (price > weekly EMA50) AND volume confirmation
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and rsi[i] > 50 and close[i] > ema_50_1w_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Price crosses below KAMA AND RSI < 50 AND weekly downtrend (price < weekly EMA50) AND volume confirmation
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and rsi[i] < 50 and close[i] < ema_50_1w_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below KAMA or trend turns bearish
            if close[i] < kama[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above KAMA or trend turns bullish
            if close[i] > kama[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals