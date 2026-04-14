#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA with weekly trend filter and volume confirmation
# KAMA adapts to market conditions - fast in trends, slow in ranges
# Weekly trend filter ensures we only trade in direction of higher timeframe trend
# Volume > 1.3x average confirms institutional participation
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(20) for trend filter
    ema_len = 20
    if len(df_weekly) < ema_len:
        return np.zeros(n)
    
    ema_weekly = pd.Series(df_weekly['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily KAMA(14, 2, 30)
    kama_length = 14
    fast_ema = 2
    slow_ema = 30
    if len(close) < kama_length:
        return np.zeros(n)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, kama_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=float)
    kama[kama_length] = close[kama_length]
    for i in range(kama_length+1, len(close)):
        if not np.isnan(sc[i-1]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-1] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(kama_length + 5, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema_weekly_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly EMA20
        above_weekly_ema = close[i] > ema_weekly_aligned[i]
        below_weekly_ema = close[i] < ema_weekly_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price > KAMA + above weekly EMA + volume
            if (close[i] > kama[i] and 
                above_weekly_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price < KAMA + below weekly EMA + volume
            elif (close[i] < kama[i] and 
                  below_weekly_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or weekly EMA
            if close[i] < kama[i] or close[i] < ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA or weekly EMA
            if close[i] > kama[i] or close[i] > ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "daily_kama_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0