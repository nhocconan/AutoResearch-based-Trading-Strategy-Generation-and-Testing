#!/usr/bin/env python3
"""
6h_Rolling_Trend_Entropy_Regime
Hypothesis: Uses rolling entropy of price changes (5-period) to detect regime shifts - low entropy = trending (follow trend), high entropy = ranging (mean revert). Combines with 1d EMA50 trend filter and volume spike (1.5x 24-bar avg) to enter trades. Designed for low trade frequency (12-37/year) to minimize fee drift while capturing regime-adaptive moves. Works in both bull and bear by adapting to market regime.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate rolling entropy of price changes (5-period)
    returns = np.diff(np.log(close), prepend=0)
    abs_returns = np.abs(returns)
    entropy_vals = np.full(n, np.nan)
    
    for i in range(5, n):
        window = abs_returns[i-4:i+1]
        if np.sum(window) > 0:
            probs = window / np.sum(window)
            entropy_vals[i] = entropy(probs, base=2)
        else:
            entropy_vals[i] = 0
    
    # Volume confirmation: >1.5x 24-period MA (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(entropy_vals[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_24[i])
        
        # Regime detection: entropy < 0.8 = trending, entropy > 1.2 = ranging
        trending_regime = entropy_vals[i] < 0.8
        ranging_regime = entropy_vals[i] > 1.2
        
        # Entry logic: follow trend in trending regime, mean revert in ranging regime
        long_entry = trending_regime and uptrend and vol_confirm
        short_entry = trending_regime and downtrend and vol_confirm
        long_exit = ranging_regime and close[i] < ema_50_1d_aligned[i]  # mean revert to trend
        short_exit = ranging_regime and close[i] > ema_50_1d_aligned[i]
        
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

name = "6h_Rolling_Trend_Entropy_Regime"
timeframe = "6h"
leverage = 1.0