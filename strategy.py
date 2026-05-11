#!/usr/bin/env python3
"""
4h_KAMA_RSI_Trend_Follow
Hypothesis: In trending markets (1d ADX > 25), KAMA direction combined with RSI filter provides reliable trend-following entries. In ranging markets (1d ADX < 25), the strategy remains flat to avoid whipsaw. Uses volume confirmation (volume > 1.5x 20-period average) to filter false signals. Designed for 20-40 trades/year per symbol to avoid fee drag while capturing trend continuation in both bull and bear markets.
"""

name = "4h_KAMA_RSI_Trend_Follow"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX and KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d ADX for regime detection (14 period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 1d KAMA (adaptive moving average) ---
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, k=10, prepend=close_1d[:10]))
    volatility = np.sum(np.abs(np.diff(close_1d, k=1)), axis=0) if len(close_1d) > 1 else np.zeros_like(close_1d)
    # Correct calculation for volatility sum over 10 periods
    volatility = np.array([np.sum(np.abs(np.diff(close_1d[max(0,i-9):i+1]))) for i in range(len(close_1d))])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # --- 1d RSI (14 period) ---
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # --- 1d Volume Average for confirmation ---
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # --- 4h Volume Average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for ADX, KAMA, RSI, and volume averages
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(vol_avg_4h[i])):
            if position != 0:
                # Check stoploss (2.0x ATR from entry)
                atr_est = np.abs(high_4h[i] - low_4h[i])  # rough 4h ATR estimate
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine regime: ADX < 25 = range, ADX > 25 = trend
        is_range = adx_1d_aligned[i] < 25
        is_trend = adx_1d_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        if position == 0:
            # Look for entries only in trending regime with volume confirmation
            if is_trend and vol_confirm:
                # KAMA direction: price above KAMA = uptrend, below = downtrend
                # RSI filter: avoid extreme readings
                if close_4h[i] > kama_1d_aligned[i] and rsi_1d_aligned[i] < 70:
                    signals[i] = 0.25  # long
                    position = 1
                    entry_price = close_4h[i]
                elif close_4h[i] < kama_1d_aligned[i] and rsi_1d_aligned[i] > 30:
                    signals[i] = -0.25  # short
                    position = -1
                    entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if is_range:
                    # In range, exit immediately to avoid whipsaw
                    signals[i] = 0.0
                    position = 0
                else:  # is_trend
                    # In trend, exit when price crosses below KAMA or RSI > 70
                    if close_4h[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] > 70:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if is_range:
                    # In range, exit immediately to avoid whipsaw
                    signals[i] = 0.0
                    position = 0
                else:  # is_trend
                    # In trend, exit when price crosses above KAMA or RSI < 30
                    if close_4h[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] < 30:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals