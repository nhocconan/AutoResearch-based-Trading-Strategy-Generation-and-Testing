#!/usr/bin/env python3
"""
1d_WKL_Trend_Momentum
Hypothesis: Uses weekly trend direction (via EMA34) and daily momentum (RSI14) for entries.
Enters long when weekly trend up and RSI crosses above 50, short when weekly trend down and RSI crosses below 50.
Uses volume confirmation (current volume > 1.5x 20-day average) to filter false signals.
Designed for fewer trades (~15-25/year) to minimize fee drag while capturing medium-term moves.
Works in bull markets via trend following and in bear markets via counter-trend RSI reversals at extremes.
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
    volume = prices['volume'].values
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need weekly EMA + RSI warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: weekly trend up AND RSI crosses above 50 with volume
            if (close[i] > ema_34_1w_aligned[i] and 
                rsi[i-1] <= 50 and rsi[i] > 50 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: weekly trend down AND RSI crosses below 50 with volume
            elif (close[i] < ema_34_1w_aligned[i] and 
                  rsi[i-1] >= 50 and rsi[i] < 50 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: weekly trend turns down OR RSI crosses below 50
            if (close[i] < ema_34_1w_aligned[i]) or (rsi[i-1] > 50 and rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up OR RSI crosses above 50
            if (close[i] > ema_34_1w_aligned[i]) or (rsi[i-1] < 50 and rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WKL_Trend_Momentum"
timeframe = "1d"
leverage = 1.0