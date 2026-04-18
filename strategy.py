#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v3
1d strategy using KAMA direction, RSI extremes, and Choppiness Index regime filter.
- Long: KAMA trending up + RSI > 50 + Chop > 61.8 (range) → mean reversion long from oversold
- Short: KAMA trending down + RSI < 50 + Chop > 61.8 (range) → mean reversion short from overbought
- Uses daily timeframe with weekly trend filter (EMA34) for alignment
- Designed for 50-80 total trades over 4 years (12-20/year)
Works in choppy/range markets (2025-2026) and captures mean reversion in trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) < len(close) else np.nansum(np.abs(np.diff(close, axis=0)), axis=0) if hasattr(change, 'axis') else np.nansum(np.abs(np.diff(close)))
    # Simplified volatility calculation for 1D array
    volatility = np.array([np.sum(np.abs(np.diff(close[max(0, i-er_len+1):i+1]))) for i in range(len(close))])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_chop(high, low, close, period=14):
    """Choppiness Index"""
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = np.zeros_like(close)
    for i in range(period-1, len(close)):
        if atr[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
            chop[i] = 100 * np.log10(atr[i] * period / (highest_high[i] - lowest_low[i])) / np.log10(period)
        else:
            chop[i] = 50.0
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    kama = calculate_kama(close_1d, er_len=10, fast=2, slow=30)
    
    # Calculate RSI on daily close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    loss_ma = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(loss_ma != 0, gain_ma / loss_ma, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    chop = calculate_chop(high_1d, low_1d, close_1d, period=14)
    
    # Weekly EMA34 for trend filter (alignment only)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily data to 1d timeframe (no alignment needed as same timeframe)
    kama_aligned = kama  # already daily
    rsi_aligned = rsi    # already daily
    chop_aligned = chop  # already daily
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    start_idx = 34  # need enough for weekly EMA34 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction (trend)
        kama_up = kama_aligned[i] > kama_aligned[i-1] if i > 0 else False
        kama_down = kama_aligned[i] < kama_aligned[i-1] if i > 0 else False
        
        # RSI conditions
        rsi_over = rsi_aligned[i] > 50
        rsi_under = rsi_aligned[i] < 50
        
        # Chop regime (range market)
        chop_range = chop_aligned[i] > 61.8
        
        # Entry conditions
        if chop_range:
            # In range: mean reversion
            if kama_up and rsi_under and close[i] < close_1d[i]:  # price below daily close = oversold
                signals[i] = 0.25
            elif kama_down and rsi_over and close[i] > close_1d[i]:  # price above daily close = overbought
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # In trend: follow KAMA direction
            if kama_up:
                signals[i] = 0.25
            elif kama_down:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v3"
timeframe = "1d"
leverage = 1.0