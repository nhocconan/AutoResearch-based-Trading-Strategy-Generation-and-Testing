#!/usr/bin/env python3
# 12h_KAMA_Direction_RSI_Trend
# Hypothesis: 12h KAMA direction (trend) combined with RSI extremes and 1d trend filter.
# Uses KAMA for adaptive trend detection, RSI for mean-reversion entries within trend,
# and 1d EMA50 for higher timeframe trend confirmation. Designed for 10-20 trades/year
# to minimize fee drag while capturing trending moves in both bull and bear markets.

name = "12h_KAMA_Direction_RSI_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle edge cases for ER calculation
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_length] = close[er_length]  # Initialize
    
    for i in range(er_length + 1, len(close)):
        if not np.isnan(sc[i-er_length]):
            kama[i] = kama[i-1] + sc[i-er_length] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI calculation
    rsi_length = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[rsi_length] = np.mean(gain[:rsi_length])
    avg_loss[rsi_length] = np.mean(loss[:rsi_length])
    
    for i in range(rsi_length + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_length - 1) + gain[i-1]) / rsi_length
        avg_loss[i] = (avg_loss[i-1] * (rsi_length - 1) + loss[i-1]) / rsi_length
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (2-period = 1 day of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (er_length), RSI (rsi_length), EMA50 (50), vol MA (2)
    start_idx = max(er_length, rsi_length, 50) + 2
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA + RSI oversold + 1d uptrend + volume surge
            if price_above_kama and rsi_oversold and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + RSI overbought + 1d downtrend + volume surge
            elif price_below_kama and rsi_overbought and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price below KAMA or RSI overbought or trend change
                if close[i] < kama[i] or rsi[i] > 70 or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price above KAMA or RSI oversold or trend change
                if close[i] > kama[i] or rsi[i] < 30 or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals