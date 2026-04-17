#!/usr/bin/env python3
"""
6h_ImpulseSystem_Trend_Momentum
Hypothesis: On 6h, capture trend momentum using EMA alignment (8/21/55) and RSI momentum (>50 for long, <50 for short) with volume confirmation. Uses 1d trend filter (price > 200 EMA) to avoid counter-trend trades. Designed for 15-25 trades/year to minimize fee fatigue and work in bull/bear regimes via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA alignment: 8, 21, 55 on 6h
    ema8 = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema55 = pd.Series(close).ewm(span=55, min_periods=55, adjust=False).mean().values
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: current vs 20-period average
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA55 and 1d EMA200
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(ema55[i]) or
            np.isnan(rsi[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_avg20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        vol_filter = volume[i] > 1.3 * vol_avg20[i]
        
        # EMA alignment: bullish (8>21>55) or bearish (8<21<55)
        ema_bullish = ema8[i] > ema21[i] > ema55[i]
        ema_bearish = ema8[i] < ema21[i] < ema55[i]
        
        # Entry conditions
        if position == 0:
            # Long: EMA bullish + RSI > 50 + above 1d EMA200 + volume
            if ema_bullish and rsi[i] > 50 and close[i] > ema200_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: EMA bearish + RSI < 50 + below 1d EMA200 + volume
            elif ema_bearish and rsi[i] < 50 and close[i] < ema200_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when EMA alignment breaks or RSI crosses 50
        elif position == 1:
            if not ema_bullish or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if not ema_bearish or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ImpulseSystem_Trend_Momentum"
timeframe = "6h"
leverage = 1.0