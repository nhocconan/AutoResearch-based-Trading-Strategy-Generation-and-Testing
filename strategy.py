#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and session timing.
Long when 1h RSI < 30 AND 4h EMA20 > EMA50 (uptrend) AND UTC hour 08-20.
Short when 1h RSI > 70 AND 4h EMA20 < EMA50 (downtrend) AND UTC hour 08-20.
Exit when RSI returns to 50 (mean reversion) OR trend flips.
Uses 4h for trend direction to avoid counter-trend trades, 1h RSI for precise entry/exit.
Session filter reduces noise during low-liquidity hours.
Target: 80-120 total trades over 4 years (20-30/year). Discrete sizing 0.20 minimizes fee churn.
Works in bull markets (buys dips in uptrend) and bear markets (sells rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMAs for trend (20 and 50)
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        ema20 = ema20_4h_aligned[i]
        ema50 = ema50_4h_aligned[i]
        rsi_val = rsi[i]
        price = close[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND 4h uptrend (EMA20 > EMA50)
            if rsi_val < 30 and ema20 > ema50:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) AND 4h downtrend (EMA20 < EMA50)
            elif rsi_val > 70 and ema20 < ema50:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI >= 50 (mean reversion) OR trend flips to downtrend
            if rsi_val >= 50 or ema20 <= ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI <= 50 (mean reversion) OR trend flips to uptrend
            if rsi_val <= 50 or ema20 >= ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_Session"
timeframe = "1h"
leverage = 1.0