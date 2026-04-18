#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v2
1d strategy using KAMA trend direction, RSI for momentum confirmation, and Choppiness Index for regime filtering.
- Long: KAMA rising + RSI > 50 + Chop > 61.8 (range) → mean reversion long at support
- Short: KAMA falling + RSI < 50 + Chop > 61.8 (range) → mean reversion short at resistance
- Uses weekly trend filter: only take longs when weekly EMA8 > EMA34, shorts when EMA8 < EMA34
- Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
Works in ranging markets via mean reversion, avoids trends via Chop filter
"""

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
    
    # Get daily data for KAMA, RSI, Chop
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    def calculate_kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum() - np.abs(np.diff(close, prepend=close[0])).cumsum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close_1d)
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    high_low_range = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr * 14 / high_low_range) / np.log10(14)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_uptrend = ema_8_1w > ema_34_1w
    weekly_downtrend = ema_8_1w < ema_34_1w
    
    # Align all daily data to 1d timeframe (no alignment needed as we're on 1d)
    kama_rising_aligned = kama_rising
    kama_falling_aligned = kama_falling
    rsi_aligned = rsi
    chop_aligned = chop
    
    # Align weekly data to 1d
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough for weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range regime: Chop > 61.8 indicates ranging market
        range_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + range regime + weekly uptrend
            if kama_rising_aligned[i] and rsi_aligned[i] > 50 and range_regime and weekly_uptrend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 50 + range regime + weekly downtrend
            elif kama_falling_aligned[i] and rsi_aligned[i] < 50 and range_regime and weekly_downtrend_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling or RSI < 40 or trend change
            if kama_falling_aligned[i] or rsi_aligned[i] < 40 or not weekly_uptrend_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising or RSI > 60 or trend change
            if kama_rising_aligned[i] or rsi_aligned[i] > 60 or not weekly_downtrend_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0