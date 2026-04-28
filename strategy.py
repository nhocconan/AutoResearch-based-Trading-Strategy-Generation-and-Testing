#!/usr/bin/env python3
"""
4h_KAMA_Adaptive_Trend_RSI_Trend_Strength
Hypothesis: Combine Kaufman Adaptive Moving Average (KAMA) for trend direction with RSI for momentum and ADX for trend strength to capture strong trends while avoiding choppy markets. Uses 1d timeframe for trend context and volume confirmation for entry quality. Designed to work in both bull and bear markets by adapting to trend strength and requiring multiple confluence factors to reduce false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend context and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close for trend direction
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate ADX on 1d for trend strength
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI on 1d for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get volume data for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume_1d > (vol_ma_20 * 1.5)
    
    # Align all higher timeframe data to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    volume_surge_aligned = align_htf_to_ltf(prices, df_1d, volume_surge)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(volume_surge_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: KAMA trend + RSI momentum + ADX strength + volume surge
        # Long: price > KAMA (uptrend) + RSI > 50 (bullish momentum) + ADX > 25 (strong trend) + volume surge
        long_entry = (close[i] > kama_aligned[i] and 
                     rsi_aligned[i] > 50 and 
                     adx_aligned[i] > 25 and 
                     volume_surge_aligned[i])
        
        # Short: price < KAMA (downtrend) + RSI < 50 (bearish momentum) + ADX > 25 (strong trend) + volume surge
        short_entry = (close[i] < kama_aligned[i] and 
                      rsi_aligned[i] < 50 and 
                      adx_aligned[i] > 25 and 
                      volume_surge_aligned[i])
        
        # Exit on opposite KAMA cross
        long_exit = close[i] < kama_aligned[i]
        short_exit = close[i] > kama_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Adaptive_Trend_RSI_Trend_Strength"
timeframe = "4h"
leverage = 1.0