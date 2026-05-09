#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyKAMA_Trend_PriceAction"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # KAMA on weekly close
    close_1w = df_1w['close'].values
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    direction = np.abs(np.diff(close_1w, n=10, prepend=close_1w[:10]))
    volatility = np.sum(change.reshape(-1, 1), axis=1)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.066 - 0.064) + 0.064) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Price action: higher highs and higher lows for uptrend, lower lows and lower highs for downtrend
    # Using 3-day lookback for simplicity
    hh = high > np.maximum.accumulate(np.roll(high, 1))
    ll = low < np.minimum.accumulate(np.roll(low, 1))
    lh = low > np.minimum.accumulate(np.roll(low, 1))
    lh = np.where(np.roll(low, 1) > 0, lh, False)  # avoid first bar
    hl = high < np.maximum.accumulate(np.roll(high, 1))
    hl = np.where(np.roll(high, 1) > 0, hl, False)
    
    # Uptrend: HH and HL
    uptrend = hh & hl
    # Downtrend: LH and LL
    downtrend = lh & ll
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 10  # Need enough data for KAMA calculation
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        is_uptrend = uptrend[i]
        is_downtrend = downtrend[i]
        
        if position == 0:
            # Enter long: price above KAMA and uptrend
            if close[i] > kama_val and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA and downtrend
            elif close[i] < kama_val and is_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA or downtrend
            if close[i] < kama_val or is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA or uptrend
            if close[i] > kama_val or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals