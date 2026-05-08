#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_SuperTrend_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h ADX (trend strength) and Supertrend (direction)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR for ADX and Supertrend
    tr1 = np.abs(np.diff(high_12h))
    tr2 = np.abs(np.diff(low_12h))
    tr3 = np.abs(high_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Movement
    up_move = np.diff(high_12h)
    down_move = -np.diff(low_12h)
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    up_move = np.concatenate([[0], up_move])
    down_move = np.concatenate([[0], down_move])
    
    # Smoothed DM
    atr_safe = np.where(atr == 0, np.nan, atr)
    plus_dm_smooth = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_smooth / atr_safe
    minus_di = 100 * minus_dm_smooth / atr_safe
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    multiplier = 3.0
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan)
    uptrend = np.ones_like(close_12h, dtype=bool)
    
    for i in range(1, len(close_12h)):
        if np.isnan(upperband[i-1]) or np.isnan(lowerband[i-1]):
            supertrend[i] = np.nan
            continue
            
        if close_12h[i] > upperband[i-1]:
            uptrend[i] = True
        elif close_12h[i] < lowerband[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if not uptrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        supertrend[i] = lowerband[i] if uptrend[i] else upperband[i]
    
    # Align 12h indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    uptrend_aligned = align_htf_to_ltf(prices, df_12h, uptrend.astype(float))
    
    # Calculate 6h momentum: price change over 3 periods
    price_change = np.diff(close, 3)
    price_change = np.concatenate([[np.nan, np.nan, np.nan], price_change])
    mom_threshold = np.nanstd(price_change) * 0.5 if not np.isnan(np.nanstd(price_change)) else 0.01
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or np.isnan(price_change[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Strong trend + momentum confirmation
        strong_trend = adx_aligned[i] > 25
        bullish_momentum = price_change[i] > mom_threshold
        bearish_momentum = price_change[i] < -mom_threshold
        
        if position == 0:
            # Long: uptrend + strong trend + bullish momentum
            if (uptrend_aligned[i] > 0.5) and strong_trend and bullish_momentum:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + strong trend + bearish momentum
            elif (uptrend_aligned[i] < 0.5) and strong_trend and bearish_momentum:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reversal or weak trend
            if (uptrend_aligned[i] < 0.5) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or weak trend
            if (uptrend_aligned[i] > 0.5) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals