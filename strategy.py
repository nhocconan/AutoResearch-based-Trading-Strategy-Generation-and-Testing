#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Filter_Volume
Hypothesis: KAMA trend direction + RSI momentum filter + volume spike on 4h.
KAMA adapts to market noise, reducing false signals in chop. RSI filters momentum extremes.
Volume spike confirms institutional interest. Works in bull/bear via adaptive trend.
Target: 25-40 trades/year to minimize fee drag.
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
    
    # Get 4h data for KAMA calculation (using same timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average)
    close_4h = df_4h['close'].values
    # Efficiency Ratio: price change / volatility
    change = np.abs(np.diff(close_4h, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_4h, k=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1, volatility)
    er = np.concatenate([np.full(10, np.nan), change / volatility])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_4h, np.nan)
    kama[29] = close_4h[29]  # seed
    for i in range(30, len(close_4h)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to lower timeframe (though same here, for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike confirmation (2x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from KAMA
        trend_up = close[i] > kama_aligned[i]
        trend_down = close[i] < kama_aligned[i]
        
        # RSI filters: avoid overbought/oversold extremes
        rsi_ok_long = rsi[i] < 70  # not overbought
        rsi_ok_short = rsi[i] > 30  # not oversold
        
        # Entry logic
        long_entry = trend_up and rsi_ok_long and vol_confirm[i]
        short_entry = trend_down and rsi_ok_short and vol_confirm[i]
        
        # Exit logic: trend reversal or RSI extreme
        long_exit = not trend_up or rsi[i] >= 70
        short_exit = not trend_down or rsi[i] <= 30
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_RSI_Filter_Volume"
timeframe = "4h"
leverage = 1.0