#!/usr/bin/env python3
"""
1d_KAMA_Regime_ADX_v1
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) identifies trend direction, 
while ADX > 25 filters for trending markets and RSI(14) < 30/ > 70 provides mean-reversion entries 
in the trend direction. Weekly EMA34 confirms higher-timeframe trend alignment. 
Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily KAMA (ER=10, fast=2, slow=30) ===
    close = prices['close'].values
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility = np.concatenate([[0], volatility[1:]])
    
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Daily ADX(14) ===
    high = prices['high'].values
    low = prices['low'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === Daily RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        adx_val = adx[i]
        rsi_val = rsi[i]
        weekly_ema = ema_34_1w_aligned[i]
        
        # Trend and regime conditions
        is_uptrend = price > kama_val
        is_downtrend = price < kama_val
        is_trending = adx_val > 25
        weekly_bull = price > weekly_ema
        weekly_bear = price < weekly_ema
        
        if position == 0:
            # Long conditions: weekly bull + daily uptrend + trending + RSI oversold
            if weekly_bull and is_uptrend and is_trending and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short conditions: weekly bear + daily downtrend + trending + RSI overbought
            elif weekly_bear and is_downtrend and is_trending and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly bear OR RSI overbought OR trend weakens
            if weekly_bear or rsi_val > 70 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly bull OR RSI oversold OR trend weakens
            if weekly_bull or rsi_val < 30 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_ADX_v1"
timeframe = "1d"
leverage = 1.0