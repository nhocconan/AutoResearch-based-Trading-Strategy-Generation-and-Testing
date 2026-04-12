#!/usr/bin/env python3
# 6h_1d_adx_macd_trend_follow
# Hypothesis: 6-hour ADX + MACD trend following with 1d trend filter. Uses ADX > 25 to confirm trend strength and MACD histogram crossover for entry. 1d EMA50 trend filter ensures trades align with higher timeframe direction, reducing whipsaws in sideways markets. Designed for 50-150 total trades over 4 years.

name = "6h_1d_adx_macd_trend_follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ADX calculation (14-period)
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(low)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # MACD (12,26,9)
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(macd_hist[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Long entry: ADX > 25 (strong trend) + MACD hist crosses above zero + uptrend
        if adx[i] > 25 and macd_hist[i] > 0 and macd_hist[i-1] <= 0 and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: ADX > 25 + MACD hist crosses below zero + downtrend
        elif adx[i] > 25 and macd_hist[i] < 0 and macd_hist[i-1] >= 0 and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: ADX falls below 20 (trend weakening) or MACD hist reverses
        elif position == 1 and (adx[i] < 20 or (macd_hist[i] < 0 and macd_hist[i-1] >= 0)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx[i] < 20 or (macd_hist[i] > 0 and macd_hist[i-1] <= 0)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals