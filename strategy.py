#!/usr/bin/env python3
"""
6h_AdaptiveKAMA_Trend_Breakout
Hypothesis: Adaptive KAMA (Kaufman Adaptive Moving Average) on 6h adapts to market noise, making it effective in both trending (bull/bear) and ranging markets. Combine with 1d trend filter (EMA34) and volume confirmation to avoid false breakouts. Entry: KAMA crosses above/below price with volume spike and 1d trend alignment. Exit: Opposite KAMA cross or volatility contraction. Designed for 60-120 total trades over 4 years.
"""

name = "6h_AdaptiveKAMA_Trend_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 6h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 1d Volume Average for confirmation ---
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # --- Adaptive KAMA on 6h (10, 2, 30) ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Pad volatility to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                # Hold position until clear exit
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 1d average
        vol_confirm = volume[i] > 1.5 * vol_avg_1d_aligned[i]
        
        # Trend filter: price above/below 1d EMA34
        price_vs_ema = close[i] > ema34_1d_aligned[i]
        
        if position == 0:
            # Look for entries: KAMA cross with volume and trend
            if i > 0 and not np.isnan(kama[i-1]):
                # Bullish: price crosses above KAMA with volume and uptrend
                if close[i-1] <= kama[i-1] and close[i] > kama[i] and vol_confirm and price_vs_ema:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price crosses below KAMA with volume and downtrend
                elif close[i-1] >= kama[i-1] and close[i] < kama[i] and vol_confirm and not price_vs_ema:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit on opposite KAMA cross or loss of trend/volume
            if position == 1:
                # Long exit: price crosses below KAMA OR trend turns down
                if (i > 0 and not np.isnan(kama[i-1]) and 
                    close[i-1] >= kama[i-1] and close[i] < kama[i]) or not price_vs_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses above KAMA OR trend turns up
                if (i > 0 and not np.isnan(kama[i-1]) and 
                    close[i-1] <= kama[i-1] and close[i] > kama[i]) or price_vs_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals