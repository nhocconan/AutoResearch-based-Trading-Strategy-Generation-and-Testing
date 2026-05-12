#!/usr/bin/env python3
name = "6h_Adaptive_Kelly_Signal_Strength"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Data for trend and volatility ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d ATR for volatility ===
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # === 1d EMA50 for trend ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 6h price position in 1d range ===
    # Calculate 1d range for each 6h bar
    range_1d = high_1d - low_1d
    range_1d_expanded = np.repeat(range_1d, 4)  # 4x 6h bars per day
    range_1d_expanded = range_1d_expanded[:n]  # trim to match
    
    # Get the 1d low for each 6h bar
    low_1d_expanded = np.repeat(low_1d, 4)
    low_1d_expanded = low_1d_expanded[:n]
    
    # Price position in daily range (0 to 1)
    price_pos = (close - low_1d_expanded) / np.maximum(range_1d_expanded, 1e-8)
    price_pos = np.clip(price_pos, 0, 1)
    
    # === 6h RSI for momentum ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.maximum(avg_loss, 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    # === 6h volume profile ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 6 days
    vol_ratio = volume / np.maximum(vol_ma, 1e-8)
    
    # === Signal strength calculation ===
    # Component 1: Trend alignment (0 to 1)
    trend_aligned = np.where(close > ema50_1d, 1.0, 0.0)
    
    # Component 2: Momentum (RSI normalized)
    momentum = (rsi - 50) / 50  # -1 to 1
    momentum = np.clip(momentum, -1, 1)
    
    # Component 3: Volatility adjustment (inverse ATR)
    vol_component = 1.0 / (1.0 + atr_1d / np.mean(atr_1d[~np.isnan(atr_1d)])) if np.any(~np.isnan(atr_1d)) else 1.0
    
    # Component 4: Volume confirmation
    vol_confirm = np.where(vol_ratio > 1.5, 1.0, 0.0)
    
    # Component 5: Price position in daily range (mean reversion edge)
    # In ranging markets, fade extremes; in trends, follow momentum
    range_signal = 0.5 - np.abs(price_pos - 0.5)  # 0 at extremes, 0.5 at middle
    range_signal = range_signal * 2  # 0 to 1
    
    # Combine components with weights
    signal_strength = (
        0.3 * trend_aligned +
        0.2 * momentum +
        0.2 * vol_component +
        0.2 * vol_confirm +
        0.1 * range_signal
    )
    
    # Apply Kelly-inspired scaling: bet size proportional to edge
    # Only take signals when strength exceeds threshold
    signal_threshold = 0.6
    signal_strength = np.where(signal_strength > signal_threshold, signal_strength, 0)
    
    # Scale to position size (0 to 0.35)
    max_position = 0.35
    signal_size = signal_strength * max_position
    
    # Apply direction based on momentum
    signal_direction = np.where(momentum > 0, 1, -1)
    signal_size = signal_size * signal_direction
    
    # Ensure we don't exceed limits
    signal_size = np.clip(signal_size, -0.35, 0.35)
    
    # Apply minimum holding period to reduce churn
    signals = np.zeros(n)
    last_signal_change = -100
    min_hold_bars = 12  # 3 days minimum hold
    
    for i in range(100, n):
        if i - last_signal_change < min_hold_bars:
            signals[i] = signals[i-1]
            continue
            
        target_signal = signal_size[i]
        current_signal = signals[i-1] if i > 0 else 0
        
        # Only change if signal difference is significant
        if abs(target_signal - current_signal) > 0.1:
            signals[i] = target_signal
            last_signal_change = i
        else:
            signals[i] = current_signal
    
    return signals