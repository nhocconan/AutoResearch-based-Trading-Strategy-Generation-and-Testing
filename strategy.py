#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Chop_Plus
Hypothesis: Combines KAMA trend with RSI momentum and Choppiness regime filter to capture trends while avoiding whipsaws in range markets. Uses 1d EMA50 for higher timeframe trend confirmation and volume surge for entry confirmation. Designed for fewer, higher-quality trades (target: 20-40/year) to minimize fee drag and improve generalization in both bull and bear markets.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA components: Efficiency Ratio and Smoothing Constants
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2  # smoothing constant
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) - range: 0-100, >61.8 = range, <38.2 = trend
    atr_14 = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14[1:] = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where(
        (highest_high - lowest_low) != 0,
        100 * np.log10(np.sum(atr_14[1:14], axis=0) / (highest_high - lowest_low)) / np.log10(14),
        50
    )
    chop = np.concatenate([np.full(14, 50), chop])  # align length
    
    # Volume surge: current > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        trend_up = ema_50_aligned[i] < close[i]  # price above EMA50 = uptrend
        trend_down = ema_50_aligned[i] > close[i]  # price below EMA50 = downtrend
        
        # Momentum and regime
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        trending_market = chop[i] < 40  # strong trend regime
        
        # Entry conditions
        long_entry = kama_up and trend_up and rsi_bullish and trending_market and volume_surge[i]
        short_entry = kama_down and trend_down and rsi_bearish and trending_market and volume_surge[i]
        
        # Exit conditions: trend change or RSI extreme
        long_exit = not kama_up or rsi[i] > 70  # exit on trend fail or overbought
        short_exit = not kama_down or rsi[i] < 30  # exit on trend fail or oversold
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_RSI_Chop_Plus"
timeframe = "4h"
leverage = 1.0