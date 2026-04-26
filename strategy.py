#!/usr/bin/env python3
"""
6h_RSI_Extremes_1dTrend_v1
Hypothesis: Trade RSI extremes (oversold/overbought) on 6h with 1d EMA50 trend filter.
In bull markets: buy 6h RSI < 30 when 1d EMA50 uptrend, exit at RSI > 70.
In bear markets: sell 6h RSI > 70 when 1d EMA50 downtrend, exit at RSI < 30.
Volume confirmation ensures momentum participation.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d EMA(50), RSI(14), volume MA(20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        rsi_val = rsi[i]
        vol_conf = volume_confirm[i]
        trend_up = close[i] > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close[i] < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND volume confirm AND 1d uptrend
            long_signal = (rsi_val < 30) and vol_conf and trend_up
            
            # Short: RSI > 70 (overbought) AND volume confirm AND 1d downtrend
            short_signal = (rsi_val > 70) and vol_conf and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: RSI > 70 (overbought) OR 1d trend flips down
            if (rsi_val > 70) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: RSI < 30 (oversold) OR 1d trend flips up
            if (rsi_val < 30) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_RSI_Extremes_1dTrend_v1"
timeframe = "6h"
leverage = 1.0