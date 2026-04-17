#!/usr/bin/env python3
"""
4h_RSI_Trend_Filter_V1
Hypothesis: On 4h timeframe, enter long when RSI(14) > 50 with bullish momentum and volume confirmation; enter short when RSI(14) < 50 with bearish momentum and volume confirmation. Uses 1-day EMA50 as trend filter to align with higher timeframe trend. Designed for moderate trade frequency (20-50/year) to capture momentum shifts in both bull and bear markets while minimizing false signals in ranging conditions.
"""

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
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # === 1-day EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1-day volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50  # For RSI and volume average
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.3x daily average volume
        vol_filter = vol_1d_current > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Momentum filters
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI > 50 + volume filter + price above daily EMA50
            if rsi_bullish and vol_filter and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI < 50 + volume filter + price below daily EMA50
            elif rsi_bearish and vol_filter and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when RSI falls below 50 (momentum shift)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when RSI rises above 50 (momentum shift)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Trend_Filter_V1"
timeframe = "4h"
leverage = 1.0