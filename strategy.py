#!/usr/bin/env python3
"""
4h_1d_Volume_Weighted_Close_Breakout
Hypothesis: 4h breakouts above/below daily Volume Weighted Close (VWC) with volume surge and trend filter using 12h EMA(21). VWC acts as a dynamic equilibrium point. Breakouts with volume indicate institutional interest. Trend filter ensures alignment with medium-term momentum. Designed for low trade frequency (<50/year) by requiring significant volume expansion and clear trend alignment. Works in bull/bear via EMA trend filter and mean-reversion exit at daily VWC.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Volume_Weighted_Close_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Volume Weighted Close (VWC)
    vwc_1d = np.zeros_like(close_1d)
    vwc_sum = 0.0
    vol_sum = 0.0
    for i in range(len(close_1d)):
        vwc_sum += close_1d[i] * volume_1d[i]
        vol_sum += volume_1d[i]
        if vol_sum > 0:
            vwc_1d[i] = vwc_sum / vol_sum
        else:
            vwc_1d[i] = close_1d[i]  # fallback
    
    # === 12H EMA(21) FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    if len(close_12h) >= 21:
        ema_21_12h = np.zeros_like(close_12h)
        ema_21_12h[0] = close_12h[0]
        alpha = 2.0 / (21 + 1)
        for i in range(1, len(close_12h)):
            ema_21_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_21_12h[i-1]
    else:
        ema_21_12h = np.full_like(close_12h, np.nan)
    
    # Align daily and 12h data to 4h timeframe
    vwc_1d_aligned = align_htf_to_ltf(prices, df_1d, vwc_1d)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Volume average (20-period for 4h = ~1.3 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(vwc_1d_aligned[i]) or np.isnan(ema_21_12h_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2.0x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below 12h EMA(21)
        price_above_ema = close[i] > ema_21_12h_aligned[i]
        price_below_ema = close[i] < ema_21_12h_aligned[i]
        
        # Breakout entries at daily VWC with volume and trend filters
        long_setup = (close[i] > vwc_1d_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < vwc_1d_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit when price returns to daily VWC (mean reversion)
        exit_long = close[i] < vwc_1d_aligned[i]
        exit_short = close[i] > vwc_1d_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals