#!/usr/bin/env python3
"""
6h_EMA_Cross_RSI_Filter_v1
Hypothesis: 6h EMA(9)/EMA(21) crossover filtered by RSI(14) extremes and 1d trend.
Long when EMA9 > EMA21 AND RSI < 30 AND price > 1d EMA50 (uptrend filter).
Short when EMA9 < EMA21 AND RSI > 70 AND price < 1d EMA50 (downtrend filter).
Exits on opposite EMA cross or RSI returning to neutral zone (40-60).
Designed for low-frequency, high-conviction trades in both bull and bear markets via 1d trend filter.
Target: 12-37 trades/year (50-150 over 4 years) by requiring EMA alignment, RSI extremes, and trend filter.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for HTF trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate EMAs for crossover
    ema_9 = pd.Series(close).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 21 for EMA21, 14 for RSI)
    start_idx = max(21, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # EMA crossover signals
        ema_bullish = ema_9[i] > ema_21[i]
        ema_bearish = ema_9[i] < ema_21[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Entry logic
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long: EMA bullish + RSI oversold
            if ema_bullish and rsi_oversold:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long: EMA bearish OR RSI returns to neutral
            elif position == 1 and (ema_bearish or rsi_neutral):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short: EMA bearish + RSI overbought
            if ema_bearish and rsi_overbought:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short: EMA bullish OR RSI returns to neutral
            elif position == -1 and (ema_bullish or rsi_neutral):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA_Cross_RSI_Filter_v1"
timeframe = "6h"
leverage = 1.0