#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w filter and volume confirmation
# KAMA adapts to market noise, reducing false signals in choppy markets.
# Weekly trend filter ensures alignment with higher timeframe direction.
# Volume confirmation filters for institutional participation.
# Designed for low frequency: target 7-25 trades/year to minimize fee drag.
# Works in bull markets (buy in uptrend) and bear markets (sell in downtrend).

name = "1d_kama_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_1w = close_1w.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate KAMA (10-period ER, 2 and 30 for SC)
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10)).values
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(10, n):  # Start after KAMA warmup
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # KAMA direction: price above/below KAMA
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: opposite KAMA cross
        exit_long = close[i] < kama[i]
        exit_short = close[i] > kama[i]
        
        if position == 1:  # Long position
            # Exit on KAMA cross down or trend reversal
            if exit_long or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on KAMA cross up or trend reversal
            if exit_short or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price above KAMA + uptrend + volume confirmation
            if kama_up and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price below KAMA + downtrend + volume confirmation
            elif kama_down and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals