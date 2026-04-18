#!/usr/bin/env python3
"""
4h_1d_Keltner_Channel_Breakout_Volume
Hypothesis: Uses 1d Keltner Channel breakouts with volume confirmation and 4h EMA trend filter.
Keltner Channels (ATR-based) adapt to volatility, providing robust breakout signals in both trending and ranging markets.
Combined with volume and EMA filter, this aims to capture strong momentum moves while avoiding false breakouts.
Target: 20-30 trades/year per symbol.
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
    
    # Get 1d data for Keltner Channel calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Keltner Channels (20-period EMA, 2.0 ATR multiplier)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period EMA of close
    ema_20 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema_20[19] = np.mean(close_1d[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1d)):
            ema_20[i] = close_1d[i] * alpha + ema_20[i-1] * (1 - alpha)
    
    # True Range and ATR (20-period)
    tr = np.zeros(len(close_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_20 = np.full(len(close_1d), np.nan)
    if len(tr) >= 20:
        atr_20[19] = np.mean(tr[0:20])
        for i in range(20, len(tr)):
            atr_20[i] = (atr_20[i-1] * 19 + tr[i]) / 20  # Wilder's smoothing
    
    # Keltner Channels
    kc_upper = ema_20 + (2.0 * atr_20)
    kc_lower = ema_20 - (2.0 * atr_20)
    
    # Get 4h data for EMA trend filter (20-period)
    ema20_4h = np.full(n, np.nan)
    if n >= 20:
        ema20_4h[19] = np.mean(close[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20_4h[i] = close[i] * alpha + ema20_4h[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 1d Keltner Channels to 4h timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or 
            np.isnan(ema20_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 1d Keltner Upper with volume spike and 4h uptrend
            if (close[i] > kc_upper_aligned[i] and vol_spike[i] and 
                close[i] > ema20_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 1d Keltner Lower with volume spike and 4h downtrend
            elif (close[i] < kc_lower_aligned[i] and vol_spike[i] and 
                  close[i] < ema20_4h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below 1d Keltner Lower or 4h trend turns down
            if (close[i] < kc_lower_aligned[i] or close[i] < ema20_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above 1d Keltner Upper or 4h trend turns up
            if (close[i] > kc_upper_aligned[i] or close[i] > ema20_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Keltner_Channel_Breakout_Volume"
timeframe = "4h"
leverage = 1.0