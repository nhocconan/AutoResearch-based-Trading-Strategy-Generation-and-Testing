#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index (< 61.8) to avoid ranging markets.
Enter long when price > KAMA, RSI > 50, and choppy regime filter off (CHOP < 61.8).
Enter short when price < KAMA, RSI < 50, and CHOP < 61.8.
Exit on opposite signal or when CHOP >= 61.8 (range regime).
Position size: 0.25 to limit drawdown in bear markets like 2022.
Target: 15-25 trades/year to stay well under 150-trade 1d hard max.
Works in bull (trends with momentum) and bear (avoids whipsaws via chop filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF regime filter (optional, can be removed if not needed)
    # Using 1w for stronger regime filter: only trade when weekly trend is aligned
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate KAMA(10, 2, 30) on daily close
    # Efficiency ratio (ER) = |change over 10 periods| / sum of absolute changes over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # needs rolling sum
    # Fix: compute volatility properly
    volatility = pd.Series(np.abs(np.diff(close))).rolling(window=10, min_periods=1).sum().values
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align with change
    er = np.where(volatility != 0, change / volatility, 0)
    er = np.concatenate([np.full(9, np.nan), er])  # first 9 values NaN
    
    # Smoothing constants: fastest EMA=2, slowest=30
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # = [ER*(fastest_sc - slowest_sc) + slowest_sc]^2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after first ER can be calculated
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])  # align
    
    # Calculate Choppiness Index(14)
    atr_period = 14
    chop_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values / hl_range) / np.log10(chop_period)
    chop = np.where(np.isnan(chop), 50.0, chop)  # default to neutral if not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), RSI (14), CHOP (14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA20)
        htf_1w_bullish = close[i] > ema_20_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_20_1w_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        is_trending = chop[i] < 61.8
        
        if position == 0:
            # Long setup: price > KAMA + RSI > 50 + 1w uptrend + trending regime
            long_setup = (close[i] > kama[i]) and (rsi[i] > 50) and htf_1w_bullish and is_trending
            
            # Short setup: price < KAMA + RSI < 50 + 1w downtrend + trending regime
            short_setup = (close[i] < kama[i]) and (rsi[i] < 50) and htf_1w_bearish and is_trending
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price <= KAMA OR RSI < 50 OR 1w trend turns bearish OR regime turns choppy
            if (close[i] <= kama[i]) or (rsi[i] < 50) or (not htf_1w_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price >= KAMA OR RSI > 50 OR 1w trend turns bullish OR regime turns choppy
            if (close[i] >= kama[i]) or (rsi[i] > 50) or (htf_1w_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0