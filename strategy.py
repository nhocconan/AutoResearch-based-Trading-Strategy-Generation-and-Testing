#!/usr/bin/env python3
"""
1d_WKLY_Trend_Breakout_With_Volume_Confirmation
Hypothesis: Weekly trend direction (using 21-period EMA) filtered by daily price action and volume confirmation.
Long when price breaks above weekly EMA21 with volume confirmation, short when breaks below with volume confirmation.
Uses 1-day ATR for volatility filter to avoid choppy markets. Designed for 1d timeframe to target 7-25 trades/year.
Works in bull markets by capturing trend continuations and in bear markets by capturing trend reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        for i in range(period, len(close)):
            ema[i] = (close[i] * 2 / (period + 1)) + ema[i-1] * (1 - 2 / (period + 1))
    return ema

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend direction
    close_1w = df_1w['close'].values
    ema_21_1w = calculate_ema(close_1w, 21)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Load daily data for entry signals and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if indicators not ready
        if np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_1d_aligned[i] > np.percentile(atr_1d_aligned[:i+1], 30) if i >= 30 else True
        
        if position == 0:
            # Long: price breaks above weekly EMA21 with volume confirmation
            if price > ema_21_1w_aligned[i] and volume_ok and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly EMA21 with volume confirmation
            elif price < ema_21_1w_aligned[i] and volume_ok and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly EMA21 (trend reversal) or volatility drops
            if price < ema_21_1w_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly EMA21 (trend reversal) or volatility drops
            if price > ema_21_1w_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WKLY_Trend_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0