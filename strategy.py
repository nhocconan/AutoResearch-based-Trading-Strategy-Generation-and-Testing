#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA trend with 1-week RSI filter and volume confirmation
# KAMA adapts to market noise, reducing whipsaw in sideways markets
# Weekly RSI > 50 filters for bullish bias, < 50 for bearish bias
# Volume confirmation ensures breakout strength
# Works in bull markets (KAMA up + RSI > 50) and bear markets (KAMA down + RSI < 50)
# Low turnover expected: ~10-20 trades/year per symbol

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE for RSI
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week RSI (14 periods)
    rsi_len = 14
    delta = np.diff(df_1w['close'].values, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_len] = np.nanmean(gain[1:rsi_len+1])
    avg_loss[rsi_len] = np.nanmean(loss[1:rsi_len+1])
    
    for i in range(rsi_len+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_len-1) + gain[i]) / rsi_len
        avg_loss[i] = (avg_loss[i-1] * (rsi_len-1) + loss[i]) / rsi_len
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Calculate KAMA (20 periods)
    kama_len = 20
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, kama_len, prepend=np.full(kama_len, np.nan)))
    volatility = np.sum(np.abs(np.diff(close, 1, prepend=np.full(1, np.nan))), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[kama_len] = close[kama_len]
    for i in range(kama_len + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, kama_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: KAMA direction
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # RSI filter: weekly RSI > 50 for long, < 50 for short
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price above KAMA + RSI bullish + volume
            if kama_up and rsi_bullish and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: price below KAMA + RSI bearish + volume
            elif kama_down and rsi_bearish and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_WeeklyRSI_Volume_v1"
timeframe = "1d"
leverage = 1.0