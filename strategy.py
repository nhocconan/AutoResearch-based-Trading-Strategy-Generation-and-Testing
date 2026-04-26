#!/usr/bin/env python3
"""
1d_WeeklyKAMA_Trend_RSIRegime_v1
Hypothesis: 1d KAMA trend direction combined with RSI regime filter and volume confirmation.
- Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
- KAMA adapts to market noise, reducing whipsaws in ranging conditions
- Long when KAMA trending up, RSI > 50 (bullish momentum), and volume above average
- Short when KAMA trending down, RSI < 50 (bearish momentum), and volume above average
- Weekly EMA34 trend filter ensures alignment with higher timeframe momentum
- Designed for low trade frequency to minimize fee drag while capturing sustained moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate KAMA on 1d
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    # Pad volatility array to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after first ER calculation
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Pad first element
    gain = np.concatenate([[np.nan], gain])
    loss = np.concatenate([[np.nan], loss])
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: above 20-period average
    df_1d_vol = get_htf_data(prices, '1d')
    volume_1d = df_1d_vol['volume'].values
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ma20 * 1.5)  # Volume at least 1.5x average
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d_vol, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for weekly EMA, 10+ for KAMA, 14 for RSI, 20 for volume MA)
    start_idx = max(34, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA trend direction
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI regime: >50 bullish, <50 bearish
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA rising AND RSI bullish AND volume spike AND weekly uptrend
            if kama_rising and rsi_bullish and volume_spike_aligned[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND RSI bearish AND volume spike AND weekly downtrend
            elif kama_falling and rsi_bearish and volume_spike_aligned[i] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA falling OR RSI turns bearish OR weekly trend turns down
            if kama_falling or not rsi_bullish or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA rising OR RSI turns bullish OR weekly trend turns up
            if kama_rising or not rsi_bearish or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyKAMA_Trend_RSIRegime_v1"
timeframe = "1d"
leverage = 1.0