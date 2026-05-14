#!/usr/bin/env python3
"""
6h_HTF_1d_1w_Camarilla_R4S4_Breakout_Volume_EMAFilter
Hypothesis: 6h Camarilla R4/S4 breakout with weekly trend filter (price > weekly EMA50 for longs, < for shorts) and volume confirmation (>1.5x 20-period volume MA). 
R4/S4 levels represent strong breakout points; weekly EMA50 filters for higher-timeframe trend alignment. 
Volume confirmation reduces false breakouts. Target 12-37 trades/year (50-150 total over 4 years).
Uses 6h primary timeframe with 1d/1w HTF for pivot calculation and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots, 1w for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla pivot points (R4, S4) from previous day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r4 = pivot + (range_ * 1.1 / 2)  # R4 = pivot + range*(1.1/2)
    s4 = pivot - (range_ * 1.1 / 2)  # S4 = pivot - range*(1.1/2)
    
    # Align R4/S4 to 6h timeframe (wait for 1d bar close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R4 + volume confirmation + weekly uptrend
            if price > r4_aligned[i] and vol_ok and price > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + volume confirmation + weekly downtrend
            elif price < s4_aligned[i] and vol_ok and price < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly EMA50 or volume confirmation fails
            if price < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly EMA50 or volume confirmation fails
            if price > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_1d_1w_Camarilla_R4S4_Breakout_Volume_EMAFilter"
timeframe = "6h"
leverage = 1.0