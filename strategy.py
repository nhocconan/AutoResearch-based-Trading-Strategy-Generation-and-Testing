#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily KAMA trend with volume confirmation and 12h RSI filter
# - Uses daily KAMA (2-period ER, 30-period SC) for trend direction
# - Uses 12h RSI(14) for overbought/oversold entry timing
# - Uses 12h volume spike for entry confirmation
# - Enters long when daily KAMA rising AND 12h RSI < 30 AND volume spike
# - Enters short when daily KAMA falling AND 12h RSI > 70 AND volume spike
# - Exits when RSI crosses 50 (mean reversion to midpoint)
# - Designed to capture mean reversion moves within the daily trend with low turnover
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1dKAMA_12hRSI_Volume"
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
    
    # Get 1d data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily KAMA (Kaufman Adaptive Moving Average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if np.sum(volatility[max(0, i-9):i+1]) > 0:
            er[i] = np.sum(change[max(0, i-9):i+1]) / np.sum(volatility[max(0, i-9):i+1])
        else:
            er[i] = 0
    # Smoothing Constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # 2-period fast, 30-period slow
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Align daily KAMA trend to 12h timeframe
    kama_rising_12h = align_htf_to_ltf(prices, df_1d, kama_rising)
    kama_falling_12h = align_htf_to_ltf(prices, df_1d, kama_falling)
    
    # RSI filter (12h timeframe)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Volume filter (12h timeframe)
    vol_ma_10 = np.convolve(volume, np.ones(10)/10, mode='same')
    vol_ma_10[:10] = np.mean(volume[:10])
    volume_spike = volume > (1.5 * vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_rising_12h[i]) or np.isnan(kama_falling_12h[i]) or 
            np.isnan(rsi_oversold[i]) or np.isnan(rsi_overbought[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI oversold, volume spike
            if kama_rising_12h[i] and rsi_oversold[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI overbought, volume spike
            elif kama_falling_12h[i] and rsi_overbought[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion)
            if rsi[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion)
            if rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals