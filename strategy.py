#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Price_Channel_Breakout
Hypothesis: KAMA adapts to market regime - fast in trends, slow in ranges. 
Breakouts above/below adaptive price channels (KAMA ± ATR) with volume confirmation capture 
trend moves while avoiding false breakouts in ranging markets. Works in both bull/bear by 
adapting speed to volatility. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA50 on 1d
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # KAMA (adaptive moving average) on 12h
    # Efficiency Ratio: |price change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # ER over 10 periods
    price_change = np.abs(close - np.roll(close, 10))
    change_sum = np.convolve(abs_change, np.ones(10), 'same')
    er = np.where(change_sum != 0, price_change / change_sum, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA2
    slow_sc = 2 / (30 + 1)  # EMA30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ATR for channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.convolve(tr, np.ones(14), 'valid') / 14
    atr = np.concatenate([np.full(13, np.nan), atr])  # align
    
    # Price channels: KAMA ± 1.5 * ATR
    upper_channel = kama + 1.5 * atr
    lower_channel = kama - 1.5 * atr
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = np.convolve(volume, np.ones(20), 'same') / 20
    vol_ma[:10] = np.nan  # insufficient lookback
    vol_ma[-10:] = np.nan  # insufficient lookahead
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 30)  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        ema50_1d = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume in uptrend (price > daily EMA50)
            if (price > upper and 
                vol_spike and 
                price > ema50_1d):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume in downtrend (price < daily EMA50)
            elif (price < lower and 
                  vol_spike and 
                  price < ema50_1d):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA or trend reverses
            if price < kama_val or price < ema50_1d:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA or trend reverses
            if price > kama_val or price > ema50_1d:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_With_Price_Channel_Breakout"
timeframe = "12h"
leverage = 1.0