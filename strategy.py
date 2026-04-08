#!/usr/bin/env python3
# 12h_1d_kama_rsi_volume_trend_v1
# Hypothesis: 12h KAMA trend direction combined with RSI momentum and volume confirmation.
# Long: KAMA rising (bullish trend), RSI > 50 (bullish momentum), volume > 1.5x 20-period average
# Short: KAMA falling (bearish trend), RSI < 50 (bearish momentum), volume > 1.5x 20-period average
# Exit: Opposite KAMA direction or RSI crosses 50
# Designed to capture medium-term trends with volume confirmation to avoid false breakouts.
# KAMA adapts to market noise, reducing whipsaws in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 12h timeframe
    def calculate_kama(price, er_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(price, n=er_period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[er_period:] = change[er_period-1:] / np.maximum(volatility[er_period-1:], 1e-10)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    
    # RSI (14-period) - 12h timeframe
    def calculate_rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    rsi_bullish = rsi > 50
    rsi_bearish = rsi < 50
    
    # Volume confirmation - 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_confirmed = volume > (1.5 * vol_ma)
    
    # Daily trend filter from 1D timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros_like(close_1d)
    for i in range(50, len(close_1d)):
        ema_50_1d[i] = np.mean(close_1d[i-50:i]) if i >= 50 else close_1d[i]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily trend: price above/below EMA50
    daily_uptrend = close > ema_50_1d_aligned
    daily_downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: KAMA falling OR RSI < 50 OR daily downtrend
            if kama_falling[i] or rsi_bearish[i] or daily_downtrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: KAMA rising OR RSI > 50 OR daily uptrend
            if kama_rising[i] or rsi_bullish[i] or daily_uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: KAMA rising AND RSI > 50 AND volume confirmed AND daily uptrend
            if (kama_rising[i] and rsi_bullish[i] and volume_confirmed[i] and daily_uptrend[i]):
                position = 1
                signals[i] = 0.25
            # Entry conditions: KAMA falling AND RSI < 50 AND volume confirmed AND daily downtrend
            elif (kama_falling[i] and rsi_bearish[i] and volume_confirmed[i] and daily_downtrend[i]):
                position = -1
                signals[i] = -0.25
    
    return signals