#!/usr/bin/env python3
"""
6h_1d_1w_Bollinger_Bandwidth_Reversion
Hypothesis: Mean reversion during low volatility contractions (squeeze) on 6h, 
filtered by 1d trend and 1w momentum. Long when price touches lower BB during 
squeeze + 1d uptrend + 1w RSI > 50. Short when price touches upper BB during 
squeeze + 1d downtrend + 1w RSI < 50. Bollinger Bandwidth identifies low-volatility 
periods prone to mean-reversion bounces. Works in ranging markets (2025) and 
catching retracements in trends. Targets 20-40 trades/year.
"""

name = "6h_1d_1w_Bollinger_Bandwidth_Reversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and 1w data for filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 1w Momentum Filter: RSI14 ---
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # --- Bollinger Bands on 6h (20, 2) ---
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    
    # --- Bollinger Bandwidth: (upper - lower) / sma ---
    bandwidth = (upper - lower) / sma
    # Bandwidth percentile rank over 50 periods to identify squeeze
    bandwidth_rank = pd.Series(bandwidth).rolling(window=50, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    # Squeeze when bandwidth is in lowest 20% percentile
    squeeze = bandwidth_rank < 0.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(bandwidth_rank[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend and 1w momentum
        trend_up = close_6h[i] > ema50_1d_aligned[i]
        trend_down = close_6h[i] < ema50_1d_aligned[i]
        mom_up = rsi_1w_aligned[i] > 50
        mom_down = rsi_1w_aligned[i] < 50
        
        if position == 0:
            # Look for mean reversion entries during squeeze
            if squeeze[i]:
                # Long: price at lower BB + 1d uptrend + 1w bullish momentum
                if close_6h[i] <= lower[i] and trend_up and mom_up:
                    signals[i] = 0.25
                    position = 1
                # Short: price at upper BB + 1d downtrend + 1w bearish momentum
                elif close_6h[i] >= upper[i] and trend_down and mom_down:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions: mean reversion complete or squeeze ends
            if position == 1:
                # Exit long: price reaches middle band OR squeeze ends
                if close_6h[i] >= sma[i] or not squeeze[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches middle band OR squeeze ends
                if close_6h[i] <= sma[i] or not squeeze[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals