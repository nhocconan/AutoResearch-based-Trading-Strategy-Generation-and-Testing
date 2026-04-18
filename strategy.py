#!/usr/bin/env python3
"""
4h CBOE Volatility Index (VIX) Inspired Fear & Greed Index for Crypto
Hypothesis: Crypto markets exhibit mean-reverting behavior in extreme fear/greed conditions.
Uses a custom Fear & Greed index based on volatility expansion/contraction, RSI extremes,
and volume imbalance to identify overextended moves. Long in extreme fear (index < 20),
short in extreme greed (index > 80). Works in both bull and bear markets by fading extremes.
Low frequency (~20-30/year) minimizes fee drag while capturing mean reversion moves.
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
    
    # Calculate volatility-based fear & greed index components
    # 1. Volatility expansion/contraction (VIX-like)
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    vol_short = pd.Series(returns).rolling(window=7, min_periods=7).std().values
    vol_long = pd.Series(returns).rolling(window=30, min_periods=30).std().values
    vol_ratio = vol_short / (vol_long + 1e-10)  # Avoid division by zero
    
    # 2. RSI extremes (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 3. Volume imbalance (buying vs selling pressure)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma + 1e-10)
    price_change = np.diff(close, prepend=close[0])
    vol_weighted_change = price_change * volume
    vol_cum = pd.Series(vol_weighted_change).rolling(window=14, min_periods=14).sum().values
    price_change_abs = np.abs(price_change)
    vol_weighted_abs = price_change_abs * volume
    vol_abs_cum = pd.Series(vol_weighted_abs).rolling(window=14, min_periods=14).sum().values
    volume_imbalance = vol_cum / (vol_abs_cum + 1e-10)  # -1 to 1 range
    
    # Normalize components to 0-100 scale
    # Volatility: high vol = fear (low index), low vol = greed (high index)
    vol_percentile = pd.Series(vol_ratio).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) > 0 else 50, raw=False
    ).values
    vol_score = 100 - (pd.Series(vol_ratio).rolling(window=50, min_periods=20).apply(
        lambda x: (np.sum(np.array(x) <= vol_ratio[i]) / len(x) * 100) if len(x) > 0 else 50,
        raw=False
    ).values)
    
    # RSI: RSI < 30 = fear, RSI > 70 = greed
    rsi_score = np.where(rsi < 30, 100 * (30 - rsi) / 30,
                        np.where(rsi > 70, 100 * (rsi - 70) / 30, 50))
    
    # Volume imbalance: negative = selling pressure (fear), positive = buying pressure (greed)
    vol_imb_score = 50 - volume_imbalance * 50  # Convert -1,1 to 0,100
    
    # Combine into Fear & Greed Index (0-100, where 0=extreme fear, 100=extreme greed)
    fear_greed = (vol_score * 0.4 + rsi_score * 0.3 + vol_imb_score * 0.3)
    fear_greed = np.clip(fear_greed, 0, 100)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        fg = fear_greed[i]
        
        if position == 0:
            # Enter long in extreme fear, short in extreme greed
            if fg < 20:  # Extreme fear
                signals[i] = 0.25
                position = 1
            elif fg > 80:  # Extreme greed
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when fear subsides (index > 40) or extreme greed
            if fg > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when greed subsides (index < 60) or extreme fear
            if fg < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Fear_Greed_Index_Mean_Reversion"
timeframe = "4h"
leverage = 1.0