#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: KAMA adapts to market noise, providing reliable trend direction. 
Combined with RSI momentum filter and Choppiness Index regime filter (avoid ranging markets), 
this strategy captures strong trends while avoiding whipsaws. 
Weekly trend filter (price vs 1w EMA50) ensures alignment with higher timeframe trend. 
Designed for low trade frequency (target: 30-100 trades over 4 years) to minimize fee drag. 
Works in both bull and bear markets via trend-following logic with regime filters.
"""

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
    
    # KAMA components
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[0:er_period] = 0
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Proper volatility calculation (sum of absolute changes)
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= er_period:
            volatility[i] -= np.abs(close[i-er_period] - close[i-er_period-1]) if i-er_period-1 >= 0 else 0
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.where(np.isnan(sc), 0, sc)
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI
    rsi_period = 14
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Choppiness Index calculation
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr1 = np.maximum(tr1, np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to primary timeframe (1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA already calculated on 1d data
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)    # RSI already calculated on 1d data
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need KAMA (10), RSI (14), chop (14), EMA50 (50)
    start_idx = max(10, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema50 = ema50_1w_aligned[i]
        
        # Trend direction: price vs KAMA
        uptrend = close_val > kama_val
        downtrend = close_val < kama_val
        
        # RSI filters: avoid extreme overbought/oversold
        rsi_not_overbought = rsi_val < 70
        rsi_not_oversold = rsi_val > 30
        
        # Chop filter: avoid ranging markets
        chop_not_extreme = chop_val < 61.8  # Not strongly ranging
        
        if position == 0:
            # Long conditions: uptrend + RSI not overbought + chop not extreme + weekly uptrend
            if uptrend and rsi_not_overbought and chop_not_extreme and close_val > ema50:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short conditions: downtrend + RSI not oversold + chop not extreme + weekly downtrend
            elif downtrend and rsi_not_oversold and chop_not_extreme and close_val < ema50:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions: trend reversal or chop extreme
            if not uptrend or chop_val >= 61.8:  # trend change or ranging market
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: trend reversal or chop extreme
            if not downtrend or chop_val >= 61.8:  # trend change or ranging market
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0