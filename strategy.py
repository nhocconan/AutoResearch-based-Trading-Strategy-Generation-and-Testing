# 12h_KAMA_Trend_RSI_Filter
# Hypothesis: Use Kaufman's Adaptive Moving Average (KAMA) on 12h for trend direction,
# filtered by RSI(14) to avoid overextended moves and volume confirmation.
# Enter long when KAMA slope positive and RSI < 50, short when KAMA slope negative and RSI > 50.
# Exit on KAMA slope reversal. Designed for low frequency (10-30 trades/year) to avoid fee drag.
# Works in bull (captures trends) and bear (avoids whipsaws via RSI filter).

name = "12h_KAMA_Trend_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman's Adaptive Moving Average (KAMA).
    Returns KAMA array.
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # For rolling volatility sum
    volatility_sum = np.zeros(n)
    for i in range(n):
        if i < 1:
            volatility_sum[i] = 0
        else:
            volatility_sum[i] = volatility_sum[i-1] + np.abs(close[i] - close[i-1])
            if i >= period:
                volatility_sum[i] -= np.abs(close[i-period] - close[i-period-1])
    
    # Avoid division by zero
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI).
    Returns RSI array.
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.prepend(delta, np.nan)  # align length
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gain and loss
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    for i in range(n):
        if i < period:
            avg_gain[i] = np.nanmean(gain[max(0, i-period+1):i+1]) if not np.isnan(gain[max(0, i-period+1):i+1]).all() else 0
            avg_loss[i] = np.nanmean(loss[max(0, i-period+1):i+1]) if not np.isnan(loss[max(0, i-period+1):i+1]).all() else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    # Avoid division by zero
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for additional context (optional)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on 12h data
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # Calculate RSI on 12h data
    rsi = calculate_rsi(close, period=14)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]) or i < 1):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # KAMA slope: positive if current > previous
        kama_slope = kama[i] - kama[i-1]
        kama_up = kama_slope > 0
        kama_down = kama_slope < 0
        
        # RSI filter: avoid overextended
        rsi_not_overbought = rsi[i] < 60
        rsi_not_oversold = rsi[i] > 40
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: KAMA slope up, RSI not overbought, volume confirmation
            if kama_up and rsi_not_overbought and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA slope down, RSI not oversold, volume confirmation
            elif kama_down and rsi_not_oversold and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: KAMA slope down or RSI overextended
            if not kama_up or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA slope up or RSI overextended
            if not kama_down or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals