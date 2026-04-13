#!/usr/bin/env python3
"""
12h_1w_1d_KAMA_Trend_with_Core_Momentum
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Combined with weekly trend filter (EMA100) and daily momentum (RSI>50 for long, RSI<50 for short),
this captures sustained trends while avoiding whipsaws in ranging markets.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Get daily data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    rsi_14_1d = calculate_rsi(close_1d, 14)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate KAMA on 12h data
    er = calculate_efficiency_ratio(close, 10)
    sc = np.square(er * (2/(2+1) - 1/(30+1)) + 1/(30+1))
    kama = calculate_kama(close, sc)
    
    # Volume confirmation: current volume > 1.3x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(kama[i]) or np.isnan(ema_100_1w_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Close above KAMA (uptrend signal)
        # 2. Price above weekly EMA100 (1w trend filter)
        # 3. RSI > 50 (bullish momentum)
        # 4. Volume expansion
        price_above_kama = close[i] > kama[i]
        price_above_weekly_ema = close[i] > ema_100_1w_aligned[i]
        bullish_momentum = rsi_14_1d_aligned[i] > 50
        long_condition = price_above_kama and price_above_weekly_ema and bullish_momentum and volume_expansion[i]
        
        # Short conditions:
        # 1. Close below KAMA (downtrend signal)
        # 2. Price below weekly EMA100 (1w trend filter)
        # 3. RSI < 50 (bearish momentum)
        # 4. Volume expansion
        price_below_kama = close[i] < kama[i]
        price_below_weekly_ema = close[i] < ema_100_1w_aligned[i]
        bearish_momentum = rsi_14_1d_aligned[i] < 50
        short_condition = price_below_kama and price_below_weekly_ema and bearish_momentum and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

def calculate_rsi(prices, period):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_efficiency_ratio(prices, period):
    change = np.abs(np.diff(prices, period))
    volatility = np.sum(np.abs(np.diff(prices)), axis=1)
    
    # Pad arrays to match length
    change_padded = np.full(len(prices), np.nan)
    volatility_padded = np.full(len(prices), np.nan)
    
    change_padded[period:] = change
    for i in range(period, len(prices)):
        volatility_padded[i] = volatility[i-period]
    
    er = np.divide(change_padded, volatility_padded, out=np.zeros_like(change_padded), where=volatility_padded!=0)
    return er

def calculate_kama(prices, sc):
    kama = np.zeros_like(prices)
    kama[0] = prices[0]
    
    for i in range(1, len(prices)):
        kama[i] = kama[i-1] + sc[i] * (prices[i] - kama[i-1])
    
    return kama

name = "12h_1w_1d_KAMA_Trend_with_Core_Momentum"
timeframe = "12h"
leverage = 1.0