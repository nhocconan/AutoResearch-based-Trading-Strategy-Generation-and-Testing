#!/usr/bin/env python3
"""
4h_Momentum_Reversal_With_Trend_Filter
Hypothesis: In BTC/ETH, momentum reversals occur when price deviates from short-term momentum while 4h trend remains intact.
Buy when RSI(14) < 30 and price > EMA(50) (oversold in uptrend). Sell when RSI(14) > 70 and price < EMA(50) (overbought in downtrend).
Uses 1d ADX(14) > 20 to filter for trending markets only. Designed for 4h to capture medium-term reversals with low trade frequency.
Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets.
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
    
    # === 4h indicators (primary timeframe) ===
    # RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # EMA(50) for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # === 1d ADX for trend strength filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / np.where(tr14 == 0, 1e-10, tr14)
    minus_di = 100 * minus_dm14 / np.where(tr14 == 0, 1e-10, tr14)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-10, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: covers RSI, EMA50, and ADX calculations
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: only trade in trending markets (ADX > 20)
        trending = adx_aligned[i] > 20
        
        if position == 0 and trending:
            # Long setup: oversold RSI in uptrend (price above EMA50)
            if rsi[i] < 30 and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short setup: overbought RSI in downtrend (price below EMA50)
            if rsi[i] > 70 and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: RSI returns to neutral or trend weakens
        elif position == 1:
            # Exit long when RSI returns to neutral (50) or trend weakens
            if rsi[i] > 50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when RSI returns to neutral (50) or trend weakens
            if rsi[i] < 50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Momentum_Reversal_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0