#!/usr/bin/env python3
"""
6h_HTF_1d_1w_Camarilla_R1S1_Breakout_Volume_EMAFilter
Hypothesis: 6h Camarilla R1/S1 breakout with volume confirmation and 1d/1w EMA trend filter. 
In 1d/1w uptrend (price > EMA34): long R1 breakout, short S1 breakdown. 
In 1d/1w downtrend (price < EMA34): short R1 breakdown, long S1 bounce (counter-trend fade). 
Volume confirmation (>1.5x 20-period volume MA) reduces false breakouts. 
Position size 0.25 balances risk/return. Target ~12-37 trades/year per symbol.
Uses 6h primary timeframe with 1d and 1w HTF for trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d and 1w for trend filter)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 34 or len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1d and 1w EMA trend filter ===
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # EMA34 on 1d and 1w
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Camarilla pivot points (R1, S1) from previous day
    # Using previous 6h bar's high, low, close (shifted by 1)
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) 
            or np.isnan(r1[i]) or np.isnan(s1[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Determine trend from 1d and 1w EMA
        trend_1d = price > ema_1d_aligned[i]
        trend_1w = price > ema_1w_aligned[i]
        # Require both timeframes to agree on trend
        uptrend = trend_1d and trend_1w
        downtrend = not trend_1d and not trend_1w
        
        if position == 0:
            if uptrend:
                # In uptrend: follow breakout
                # Long: price breaks above R1 + volume confirmation
                if price > r1[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: price breaks below S1 + volume confirmation
                elif price < s1[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif downtrend:
                # In downtrend: fade extremes (counter-trend)
                # Short: price rejects at R1 + volume confirmation
                if price > r1[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                # Long: price bounces off S1 + volume confirmation
                elif price < s1[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
        
        elif position == 1:
            # Exit on opposite signal or volume drying up
            if (downtrend and price < s1[i]) or (not vol_ok and i > 0 and volume_6h[i-1] > 1.5 * vol_ma[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on opposite signal or volume drying up
            if (uptrend and price > r1[i]) or (not vol_ok and i > 0 and volume_6h[i-1] > 1.5 * vol_ma[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_1d_1w_Camarilla_R1S1_Breakout_Volume_EMAFilter"
timeframe = "6h"
leverage = 1.0