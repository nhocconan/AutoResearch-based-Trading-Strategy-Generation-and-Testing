#!/usr/bin/env python3
"""
1d_RSI_VWAP_Reversion_v1
Daily RSI(14) mean reversion with VWAP filter and weekly trend filter.
- Long: RSI < 30, price > VWAP, price above weekly EMA50
- Short: RSI > 70, price < VWAP, price below weekly EMA50
- Exit: RSI crosses back to 50 or weekly trend changes
Designed to work in both bull and bear markets by fading extremes in direction of weekly trend.
Target: 50-100 total trades over 4 years (12-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === VWAP (daily reset) ===
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.zeros(n)
    cum_vol = np.zeros(n)
    running_tpv = 0
    running_vol = 0
    for i in range(n):
        running_tpv += tpv[i]
        running_vol += volume[i]
        cum_tpv[i] = running_tpv
        cum_vol[i] = running_vol
    vwap = cum_tpv / (cum_vol + 1e-10)
    
    # === Weekly EMA50 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI < 30, price > VWAP, price above weekly EMA50
            if (rsi[i] < 30 and 
                close[i] > vwap[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI > 70, price < VWAP, price below weekly EMA50
            elif (rsi[i] > 70 and 
                  close[i] < vwap[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI > 50 OR price below weekly EMA50 (trend change)
            if (rsi[i] > 50 or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 OR price above weekly EMA50 (trend change)
            if (rsi[i] < 50 or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_VWAP_Reversion_v1"
timeframe = "1d"
leverage = 1.0