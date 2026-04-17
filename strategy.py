#!/usr/bin/env python3
"""
12h_WMA_Cross_Volume_Momentum_v1
12-hour strategy using 50/200 period Weighted Moving Average crossover with volume and momentum confirmation.
Enters long when WMA50 crosses above WMA200 with volume above average and RSI > 50.
Enters short when WMA50 crosses below WMA200 with volume above average and RSI < 50.
Uses 1-week trend filter to align with higher timeframe momentum.
Designed for low trade frequency to minimize fee drag while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1-week Trend Filter (EMA200) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === 12h WMA50 and WMA200 ===
    wma50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    wma200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === 12h Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h RSI(14) for Momentum ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with 50 (neutral)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(wma50[i]) or np.isnan(wma200[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # WMA crossover signals
        wma50_prev = wma50[i-1]
        wma200_prev = wma200[i-1]
        cross_up = wma50[i] > wma200[i] and wma50_prev <= wma200_prev
        cross_down = wma50[i] < wma200[i] and wma50_prev >= wma200_prev
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # 1-week trend filter
        uptrend_1w = close[i] > ema200_1w_aligned[i]
        downtrend_1w = close[i] < ema200_1w_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: WMA50 crosses above WMA200, volume confirmed, RSI > 50, 1w uptrend
            if cross_up and vol_confirmed and rsi[i] > 50 and uptrend_1w:
                signals[i] = 0.25
                position = 1
                continue
            # Short: WMA50 crosses below WMA200, volume confirmed, RSI < 50, 1w downtrend
            elif cross_down and vol_confirmed and rsi[i] < 50 and downtrend_1w:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse crossover
        elif position == 1:
            # Exit long: WMA50 crosses below WMA200
            if cross_down:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: WMA50 crosses above WMA200
            if cross_up:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WMA_Cross_Volume_Momentum_v1"
timeframe = "12h"
leverage = 1.0