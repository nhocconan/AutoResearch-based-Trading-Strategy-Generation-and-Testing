#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter_v1
# Hypothesis: KAMA adapts to market noise, providing a smooth trend filter. RSI(14) identifies momentum extremes. Choppiness index filters ranging markets. This combination reduces whipsaws in both bull and bear regimes by only taking trades when trend is clear (KAMA slope) and momentum is aligned (RSI), while avoiding choppy markets. Designed for low trade frequency (<25/year) to minimize fee drag.

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    kama_period = 10
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio and Smoothing Constant for KAMA
    change = np.abs(np.diff(close, kama_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[kama_period] = close[kama_period]
    for i in range(kama_period + 1, n):
        if not np.isnan(sc[i - kama_period - 1]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i - kama_period - 1] * (close[i] - kama[i-1])
    
    # Align weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # RSI(14) calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = np.zeros_like(close)
    for i in range(atr_period, n):
        if atr[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
            chop[i] = 100 * np.log10(np.sum(tr[i-atr_period+1:i+1]) / (atr[i] * atr_period)) / np.log10(atr_period)
        else:
            chop[i] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), Chop (14), weekly EMA (34)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend filter: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum: avoid extremes, look for momentum in direction of trend
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        rsi_bullish = 50 < rsi[i] < 70
        rsi_bearish = 30 < rsi[i] < 50
        
        # Chop filter: only trade in trending markets (CHOP < 61.8)
        trending_market = chop[i] < 61.8
        
        if position == 0:
            # Long entry: price above KAMA (uptrend), RSI bullish momentum, trending market
            if price_above_kama and rsi_bullish and trending_market:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA (downtrend), RSI bearish momentum, trending market
            elif price_below_kama and rsi_bearish and trending_market:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI becomes overbought
            if not price_above_kama or rsi_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI becomes oversold
            if not price_below_kama or rsi_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals