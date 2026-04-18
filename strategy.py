#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_Volume_Trend
Hypothesis: Daily breakouts above Keltner upper band or below lower band with volume confirmation and weekly EMA trend filter.
Designed for low trade frequency (target: 15-25/year) with strong performance in both bull and bear markets.
Uses Keltner channels (ATR-based) for dynamic support/resistance and volume filter to avoid false breakouts.
Weekly EMA ensures we only trade with the higher timeframe trend.
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
    
    # Calculate 1-day Keltner Channels (20-period EMA + 2*ATR)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(20)
    atr = np.full(n, np.nan)
    for i in range(20, n):
        if i == 20:
            atr[i] = np.nanmean(tr[1:21])  # Skip first NaN
        else:
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # EMA(20) of close
    ema_close = np.full(n, np.nan)
    k = 2 / (20 + 1)
    for i in range(n):
        if i == 0:
            ema_close[i] = close[i]
        else:
            ema_close[i] = close[i] * k + ema_close[i-1] * (1 - k)
    
    # Keltner Bands
    keltner_upper = ema_close + 2 * atr
    keltner_lower = ema_close - 2 * atr
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Get weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    k_w = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema50_1w[i] = close_1w[i]
        else:
            ema50_1w[i] = close_1w[i] * k_w + ema50_1w[i-1] * (1 - k_w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Keltner band with volume spike and weekly uptrend
            if (close[i] > keltner_upper[i] and vol_spike[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner band with volume spike and weekly downtrend
            elif (close[i] < keltner_lower[i] and vol_spike[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below EMA(20) or weekly trend turns down
            if (close[i] < ema_close[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above EMA(20) or weekly trend turns up
            if (close[i] > ema_close[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Channel_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0