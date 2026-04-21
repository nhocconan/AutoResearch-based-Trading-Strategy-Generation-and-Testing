#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 12-hour RSI divergence with 1-day ADX trend filter.
Look for bullish/bearish RSI divergence on 12h chart (price makes new low/high but RSI does not)
combined with strong trend (ADX > 25) on daily timeframe. Enter on RSI reversal confirmation.
Works in trending markets (both bull and bear) by fading momentum exhaustion. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high, low, close, period=14):
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE for RSI divergence
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h RSI
    rsi_12h = calculate_rsi(df_12h['close'].values, 14)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 12h price highs/lows for divergence detection
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    
    # Load 1d data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(high_12h_aligned[i]) or 
            np.isnan(low_12h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        rsi_now = rsi_12h_aligned[i]
        rsi_prev = rsi_12h_aligned[i-1] if i > 0 else rsi_now
        high_now = high_12h_aligned[i]
        high_prev = high_12h_aligned[i-1] if i > 0 else high_now
        low_now = low_12h_aligned[i]
        low_prev = low_12h_aligned[i-1] if i > 0 else low_now
        adx_val = adx_1d_aligned[i]
        
        # Detect RSI divergence (need at least 3 bars to compare)
        if i >= 3:
            # Bullish divergence: price makes lower low but RSI makes higher low
            bull_div = (low_now < low_prev and 
                       rsi_now > rsi_12h_aligned[i-2])
            # Bearish divergence: price makes higher high but RSI makes lower high
            bear_div = (high_now > high_prev and 
                       rsi_now < rsi_12h_aligned[i-2])
        else:
            bull_div = bear_div = False
        
        if position == 0:
            # Enter long: bullish RSI divergence + strong uptrend (ADX > 25)
            if bull_div and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish RSI divergence + strong downtrend (ADX > 25)
            elif bear_div and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI reaches overbought/oversold or trend weakens
            if position == 1 and (rsi_now > 70 or adx_val < 20):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_now < 30 or adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_RSIDivergence_1dADX_TrendFilter"
timeframe = "6h"
leverage = 1.0