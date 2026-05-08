#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with RSI filter and volume spike confirmation.
# Long when KAMA crosses above RSI(14) midpoint (50) AND volume > 1.5x 20-period average.
# Short when KAMA crosses below RSI(14) midpoint (50) AND volume > 1.5x 20-period average.
# Exit when KAMA crosses back to the opposite side of RSI(14) midpoint.
# KAMA adapts to market noise, reducing whipsaws in ranging markets.
# RSI midpoint filter ensures trades align with momentum.
# Volume spike confirms institutional participation.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_KAMA_RSI50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1w EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Sufficient warmup for RSI and KAMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA crosses above RSI midpoint (50), uptrend, volume filter
            long_cond = (kama[i] > 50) and (kama[i-1] <= 50) and (ema20_1w_aligned[i] > ema20_1w_aligned[i-1]) and volume_filter[i]
            # Short conditions: KAMA crosses below RSI midpoint (50), downtrend, volume filter
            short_cond = (kama[i] < 50) and (kama[i-1] >= 50) and (ema20_1w_aligned[i] < ema20_1w_aligned[i-1]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA crosses below RSI midpoint (50)
            if kama[i] < 50 and kama[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA crosses above RSI midpoint (50)
            if kama[i] > 50 and kama[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals