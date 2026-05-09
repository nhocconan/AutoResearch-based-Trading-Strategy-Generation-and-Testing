#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_RSI_Trend_Filter"
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
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on daily close
    close_d = df_d['close'].values
    ema_34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_d_aligned = align_htf_to_ltf(prices, df_d, ema_34_d)
    
    # Calculate 14-period RSI on 12h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate KAMA on 12h close
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix volatility calculation - need rolling sum of absolute changes
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            volatility[i] -= np.abs(close[i-10] - close[i-11]) if i >= 11 else 0
    # Simplified volatility calculation using rolling sum
    vol_series = pd.Series(np.abs(np.diff(close, prepend=close[0])))
    volatility = vol_series.rolling(window=10, min_periods=10).sum().values
    
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_d_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(kama[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema_34_d_aligned[i]
        rsi_val = rsi[i]
        kama_val = kama[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above KAMA + RSI > 50 + daily EMA uptrend + volume filter
            if close[i] > kama_val and rsi_val > 50 and close[i] > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below KAMA + RSI < 50 + daily EMA downtrend + volume filter
            elif close[i] < kama_val and rsi_val < 50 and close[i] < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below KAMA or RSI < 40
            if close[i] < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above KAMA or RSI > 60
            if close[i] > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals