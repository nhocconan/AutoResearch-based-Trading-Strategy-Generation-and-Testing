#!/usr/bin/env python3
"""
1d_KAMA_Regime_VolumeBreakout_1wTrend
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) identifies trend direction.
Enter long when price breaks above KAMA with volume spike and weekly uptrend; short when price breaks below KAMA with volume spike and weekly downtrend.
Uses ATR-based stop loss and discrete position sizing (0.25) to limit trades (~15-25/year) and minimize fee drag.
Designed to work in both bull and bear markets by combining adaptive trend, volume confirmation, and weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily KAMA (adaptive trend)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ATR for stop loss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need KAMA (10), ATR (14), volume MA (20), aligned weekly EMA
    start_idx = max(30, 20, 14, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above KAMA with volume spike and weekly uptrend
            long_breakout = (curr_close > kama[i]) and vol_spike[i] and (curr_close > ema_34_1w_aligned[i])
            # Short: price breaks below KAMA with volume spike and weekly downtrend
            short_breakout = (curr_close < kama[i]) and vol_spike[i] and (curr_close < ema_34_1w_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below KAMA OR trend turns down OR ATR stoploss hit
            if (curr_close < kama[i]) or (curr_close < ema_34_1w_aligned[i]) or (curr_close < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above KAMA OR trend turns up OR ATR stoploss hit
            if (curr_close > kama[i]) or (curr_close > ema_34_1w_aligned[i]) or (curr_close > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Regime_VolumeBreakout_1wTrend"
timeframe = "1d"
leverage = 1.0