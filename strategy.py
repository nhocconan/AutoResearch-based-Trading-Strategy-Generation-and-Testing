#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate efficiency ratio and smoothing constant
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder - will compute properly below
    
    # Proper KAMA calculation
    lookback = 10
    change_abs = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_sum = np.convolve(np.abs(np.diff(close_1d)), np.ones(lookback), 'same')
    volatility_sum[:lookback-1] = np.sum(np.abs(np.diff(close_1d[:lookback])), axis=0)
    
    er = np.zeros_like(close_1d)
    for i in range(lookback, len(close_1d)):
        if volatility_sum[i] > 0:
            er[i] = change_abs[i] / volatility_sum[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # RSI (14) - daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Choppiness Index (14) - daily filter
    atr_period = 14
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    sum_atr = pd.Series(atr).rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(atr_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter - 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20 * 1.5
    
    # Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        price_above_weekly_sma = close[i] > sma50_1w_aligned[i]
        price_below_weekly_sma = close[i] < sma50_1w_aligned[i]
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        chop_high = chop_aligned[i] > 61.8  # ranging market
        chop_low = chop_aligned[i] < 38.2   # trending market
        
        if position == 0:
            # Long: Price above KAMA + above weekly SMA + RSI oversold + chop > 61.8 (range) + volume
            if price_above_kama and price_above_weekly_sma and rsi_oversold and chop_high and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price below KAMA + below weekly SMA + RSI overbought + chop > 61.8 (range) + volume
            elif price_below_kama and price_below_weekly_sma and rsi_overbought and chop_high and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below KAMA OR RSI overbought OR chop < 38.2 (trend change)
                if close[i] < kama_1d_aligned[i] or rsi_aligned[i] > 70 or chop_aligned[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above KAMA OR RSI oversold OR chop < 38.2 (trend change)
                if close[i] > kama_1d_aligned[i] or rsi_aligned[i] < 30 or chop_aligned[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals