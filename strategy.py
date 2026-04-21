#!/usr/bin/env python3
"""
12h_RSI2_Contrarian_With_Volume_Filter
Hypothesis: Use contrarian RSI(2) signals on 12h timeframe with volume confirmation.
Buy when RSI(2) < 10 and price > EMA200 (uptrend dip buy).
Sell when RSI(2) > 90 and price < EMA200 (downtrend rally sell).
Add volume filter to avoid false signals in low-volume periods.
Designed for low trade frequency (15-25/year) to minimize fee drag.
Works in bull markets by buying dips and in bear markets by selling rallies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=2):
    """Calculate Relative Strength Index"""
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    if len(close) >= period:
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema200_1d = calculate_ema(close_1d, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(2) on 12h closes
    rsi2 = calculate_rsi(prices['close'].values, 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi2[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Uptrend: price > daily EMA200
            if price > ema200_1d_aligned[i]:
                # Contrarian long: RSI(2) < 10 (oversold) with volume confirmation
                if rsi2[i] < 10 and volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Downtrend: price < daily EMA200
            elif price < ema200_1d_aligned[i]:
                # Contrarian short: RSI(2) > 90 (overbought) with volume confirmation
                if rsi2[i] > 90 and volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI(2) > 50 (normal) or trend change
            if rsi2[i] > 50 or price < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI(2) < 50 (normal) or trend change
            if rsi2[i] < 50 or price > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI2_Contrarian_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0