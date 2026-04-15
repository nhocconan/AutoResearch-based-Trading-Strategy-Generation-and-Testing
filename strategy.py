#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4h trend filter and 1d regime filter
# Uses RSI(2) for short-term mean reversion entries in direction of 4h EMA50 trend,
# only when 1d ADX < 25 (low volatility regime). Avoids trend-following whipsaws
# in choppy markets while capturing mean reversion in low-volatility environments.
# Target: 60-150 total trades over 4 years (15-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate RSI(2) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA50 on 4h for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ADX(14) on 1d for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 1h timeframe
    rsi_aligned = rsi  # already on 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: RSI < 10 (oversold) + price above 4h EMA50 + ADX < 25 (low volatility)
        if (rsi_aligned[i] < 10 and
            close[i] > ema50_4h_aligned[i] and
            adx_aligned[i] < 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: RSI > 90 (overbought) + price below 4h EMA50 + ADX < 25 (low volatility)
        elif (rsi_aligned[i] > 90 and
              close[i] < ema50_4h_aligned[i] and
              adx_aligned[i] < 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or RSI returns to neutral (50) or ADX > 25 (high volatility)
        elif position == 1 and (rsi_aligned[i] > 50 or adx_aligned[i] > 25):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_aligned[i] < 50 or adx_aligned[i] > 25):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSI2_4hTrend_1dADX_Filter"
timeframe = "1h"
leverage = 1.0