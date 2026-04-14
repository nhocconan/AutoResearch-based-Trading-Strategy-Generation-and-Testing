#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA Trend + RSI Mean Reversion + Volume Spike
# Uses KAMA (Kaufman Adaptive Moving Average) for trend detection, 
# RSI(14) for mean reversion signals, and volume spike confirmation.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# In trending markets (KAMA slope aligned), we look for RSI extremes to fade.
# Volume spike confirms institutional participation.
# Works in bull/bear by fading overextended moves in the direction of the trend.
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 10-period
    # Efficiency Ratio
    change = np.abs(np.diff(close, 1))
    change = np.insert(change, 0, 0)  # align with original length
    
    volatility = np.abs(np.diff(close, 1))
    volatility = np.insert(volatility, 0, 0)
    
    # Sum over 10 periods
    er_num = np.abs(np.subtract(close[9:], np.roll(close, 9)[9:]))
    er_num = np.concatenate([np.zeros(9), er_num])
    
    er_den = np.sum(np.lib.stride_tricks.sliding_window_view(volatility, 10), axis=1)
    er_den = np.concatenate([np.zeros(9), er_den])
    
    er = np.where(er_den != 0, er_num / er_den, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (trend direction)
    kama_slope = np.diff(kama, 1)
    kama_slope = np.insert(kama_slope, 0, 0)
    
    # RSI (14-period)
    delta = np.diff(close, 1)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 2.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for RSI and KAMA stability
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(rsi[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: KAMA slope indicates trend direction
        # Only trade when there's a clear trend
        if abs(kama_slope[i]) < 0.01 * close[i]:  # minimal slope threshold
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below KAMA (dip in uptrend) + RSI oversold + volume spike
            if (price < kama[i] and kama_slope[i] > 0 and 
                rsi[i] < 30 and vol > 2.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price above KAMA (pullback in downtrend) + RSI overbought + volume spike
            elif (price > kama[i] and kama_slope[i] < 0 and 
                  rsi[i] > 70 and vol > 2.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above KAMA or RSI overbought
            if price > kama[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below KAMA or RSI oversold
            if price < kama[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_KAMA_RSI_Volume_Spike"
timeframe = "4h"
leverage = 1.0