#!/usr/bin/env python3
"""
4h_KAMA_Trend_1D_TrendFilter_Volume
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) direction on 4h for trend, filtered by 1d EMA trend and volume spike. 
Go long when KAMA turns up (bullish shift) with price above 1d EMA50 and volume > 1.5x 20-bar average. 
Go short when KAMA turns down (bearish shift) with price below 1d EMA50 and volume confirmation.
Exit when KAMA reverses or price crosses 1d EMA50.
Designed for 4h timeframe to balance signal quality and trade frequency (target: 20-50 trades/year).
KAMA adapts to market noise, reducing whipsaws in chop while capturing trends.
"""

name = "4h_KAMA_Trend_1D_TrendFilter_Volume"
timeframe = "4h"
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
    
    # Get daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on 4h close
    # Efficiency ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    # Avoid index error for first 10 bars
    change[:10] = np.nan
    
    abs_diff = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(abs_diff).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smooth ER
    er = pd.Series(er).ewm(alpha=1, adjust=False).mean().values  # ER is already 0-1
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 4h (no extra delay needed as it's LTF)
    kama_4h = kama
    
    # Get daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-bar) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after KAMA warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_4h[i]) or np.isnan(kama_4h[i-1]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-bar average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: KAMA turns up (current > previous) + price above daily EMA50 + volume spike
            if kama_4h[i] > kama_4h[i-1] and close[i] > ema_50_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turns down (current < previous) + price below daily EMA50 + volume spike
            elif kama_4h[i] < kama_4h[i-1] and close[i] < ema_50_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or price crosses below daily EMA50
            if kama_4h[i] < kama_4h[i-1] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or price crosses above daily EMA50
            if kama_4h[i] > kama_4h[i-1] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals