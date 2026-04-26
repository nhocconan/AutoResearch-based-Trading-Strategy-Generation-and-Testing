#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, enter long when KAMA trend is up (price > KAMA) AND RSI(14) > 50 AND Choppiness Index(14) < 38.2 (trending regime). Enter short when KAMA trend is down (price < KAMA) AND RSI(14) < 50 AND Choppiness Index(14) < 38.2. Exit on opposite regime or extreme RSI. Uses 1-week EMA34 as higher timeframe trend filter to avoid counter-trend trades. Targets 7-25 trades/year by requiring confluence of trend, momentum, and regime filters.
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
    
    # Get 1d data for KAMA, RSI, Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate KAMA (ER=10, FAST=2, SLOW=30) on 1d close
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    volatility_series = pd.Series(volatility)
    volatility_ma = volatility_series.rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_ma > 0, change / volatility_ma, 0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast = 2.0
    slow = 30.0
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    sc = np.nan_to_num(sc, nan=0.0)
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    delta = np.concatenate([np.array([np.nan]), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss_series.rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    # Calculate Choppiness Index(14) on 1d OHLC
    atr_14 = np.zeros_like(close_1d)
    tr1 = np.abs(np.subtract(high[1:], low[:-1]))
    tr2 = np.abs(np.subtract(high[1:], close_1d[:-1]))
    tr3 = np.abs(np.subtract(low[1:], close_1d[:-1]))
    tr = np.concatenate([np.array([np.nan]), np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_series = pd.Series(tr)
    atr_14 = atr_series.rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where(
        (max_high - min_low) > 0,
        100 * np.log10(atr_14.sum() / (max_high - min_low)) / np.log10(14),
        50.0
    )
    # Handle NaN from sum
    chop_series = pd.Series(chop)
    chop = chop_series.rolling(window=14, min_periods=14).mean().values
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align 1d indicators to lower timeframe (prices index)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need all indicators ready
    start_idx = max(34, 14, 10)  # 1w EMA34, RSI14, Chop14, KAMA ER=10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade in trending market (Chop < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        # KAMA trend direction
        kama_uptrend = close[i] > kama_aligned[i]
        kama_downtrend = close[i] < kama_aligned[i]
        
        # RSI filter
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # 1w EMA34 trend filter (avoid counter-trend trades)
        w_trend_uptrend = close[i] > ema_34_1w_aligned[i]
        w_trend_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA uptrend + RSI > 50 + trending regime + 1w uptrend
            long_signal = kama_uptrend and rsi_bullish and trending_regime and w_trend_uptrend
            
            # Short: KAMA downtrend + RSI < 50 + trending regime + 1w downtrend
            short_signal = kama_downtrend and rsi_bearish and trending_regime and w_trend_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA downtrend OR RSI < 30 (oversold) OR chop > 61.8 (choppy) OR 1w trend change
            if (kama_downtrend or rsi_aligned[i] < 30 or chop_aligned[i] > 61.8 or not w_trend_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA uptrend OR RSI > 70 (overbought) OR chop > 61.8 (choppy) OR 1w trend change
            if (kama_uptrend or rsi_aligned[i] > 70 or chop_aligned[i] > 61.8 or not w_trend_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0