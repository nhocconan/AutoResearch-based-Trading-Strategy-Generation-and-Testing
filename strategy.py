#!/usr/bin/env python3
"""
12h_1d_KAMA_Direction_RSI_Trend_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a reliable trend signal.
In trending markets (ADX > 25), KAMA direction filters noise; in ranging markets (ADX < 20), RSI extremes (30/70) provide mean-reversion entries.
Combined with 1d trend filter and volume confirmation, this adapts to both bull and bear regimes.
Target: 15-35 trades/year per symbol.
"""

name = "12h_1d_KAMA_Direction_RSI_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 50 EMA (slow, reliable)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 12h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # KAMA (Kaufman Adaptive Moving Average) - 10-period ER, 2/30 SC
    def kama(close, er_len=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder
        # Proper volatility: sum of absolute changes over er_len period
        volatility = np.zeros_like(close)
        for i in range(er_len, len(close)):
            volatility[i] = np.sum(np.abs(np.diff(close[i-er_len:i+1])))
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        # Smooth ER
        sc = np.power(er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1), 2)
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close, 10, 2, 30)
    kama_dir = kama_val > np.roll(kama_val, 1)  # rising
    
    # RSI (14)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_val = rsi(close, 14)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.3 * vol_ma
    
    # ADX (14) for regime filter
    def adx(high, low, close, length=14):
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        tr = np.maximum(high - low, 
                        np.maximum(np.abs(high - np.roll(close, 1)), 
                                   np.abs(low - np.roll(close, 1))))
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values / \
                  pd.Series(tr).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values / \
                   pd.Series(tr).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx_out = pd.Series(dx).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        return adx_out
    
    adx_val = adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values for current bar
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        adx = adx_val[i]
        kama_up = kama_dir[i]
        rsi = rsi_val[i]
        
        if position == 0:
            # LONG: KAMA up AND (trending: ADX>25) OR (ranging: RSI<30) AND volume confirmation AND 1d uptrend
            if kama_up and ((adx > 25) or (rsi < 30)) and vol_conf and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down AND (trending: ADX>25) OR (ranging: RSI>70) AND volume confirmation AND 1d downtrend
            elif not kama_up and ((adx > 25) or (rsi > 70)) and vol_conf and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA down OR 1d trend turns down
            if not kama_up or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up OR 1d trend turns up
            if kama_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals