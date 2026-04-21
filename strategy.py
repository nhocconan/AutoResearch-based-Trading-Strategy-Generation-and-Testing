#!/usr/bin/env python3
"""
12h_1w_1d_Pivot_R2S2_Breakout_Volume_ATRFilter_v1
Hypothesis: Weekly and daily pivot points R2/S2 form strong weekly/daily support/resistance. 
Breakouts above R2 (long) or below S2 (short) with volume confirmation and ATR-based stops capture 
significant moves. Designed for low trade frequency (~15-30/year) to minimize fee drag in bear markets.
Uses 12h primary timeframe with 1w/1d pivot points. Weekly pivot adds higher timeframe context filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly and daily data once for pivot points and ATR
    df_weekly = get_htf_data(prices, '1w')
    df_daily = get_htf_data(prices, '1d')
    
    if len(df_weekly) < 2 or len(df_daily) < 2:
        return np.zeros(n)
    
    # Weekly pivot points (higher timeframe context)
    wh = df_weekly['high'].values
    wl = df_weekly['low'].values
    wc = df_weekly['close'].values
    wp = (wh + wl + wc) / 3.0
    wr2 = wp + (wh - wl)
    ws2 = wp - (wh - wl)
    
    # Daily pivot points (entry levels)
    dh = df_daily['high'].values
    dl = df_daily['low'].values
    dc = df_daily['close'].values
    dp = (dh + dl + dc) / 3.0
    dr2 = dp + (dh - dl)
    ds2 = dp - (dh - dl)
    
    # Daily ATR for volatility filtering and stops
    tr1 = np.abs(dh - dl)
    tr2 = np.abs(dh - np.roll(dc, 1))
    tr3 = np.abs(dl - np.roll(dc, 1))
    tr1[0] = dh[0] - dl[0]
    tr2[0] = np.abs(dh[0] - dc[0])
    tr3[0] = np.abs(dl[0] - dc[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 12h timeframe
    wr2_aligned = align_htf_to_ltf(prices, df_weekly, wr2)
    ws2_aligned = align_htf_to_ltf(prices, df_weekly, ws2)
    dr2_aligned = align_htf_to_ltf(prices, df_daily, dr2)
    ds2_aligned = align_htf_to_ltf(prices, df_daily, ds2)
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(wr2_aligned[i]) or np.isnan(ws2_aligned[i]) or 
            np.isnan(dr2_aligned[i]) or np.isnan(ds2_aligned[i]) or 
            np.isnan(atr_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_daily_aligned[i]
        wr2 = wr2_aligned[i]
        ws2 = ws2_aligned[i]
        dr2 = dr2_aligned[i]
        ds2 = ds2_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.5x 30-period average (balanced frequency)
        if i >= 30:
            vol_ma = np.mean(volume[i-30:i])
        else:
            vol_ma = volume[i] if i > 0 else 0
        vol_ok = vol_current > 2.5 * vol_ma
        
        # Weekly context filter: only take longs above weekly R2, shorts below weekly S2
        weekly_long_ok = price > wr2
        weekly_short_ok = price < ws2
        
        if position == 0:
            # Long breakout: price breaks above daily R2 with volume and weekly context
            if price > dr2 and vol_ok and weekly_long_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below daily S2 with volume and weekly context
            elif price < ds2 and vol_ok and weekly_short_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below daily S2 (failed breakout) or ATR-based stop
            if price < ds2 or (i > 0 and close[i-1] > ds2 and price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above daily R2 (failed breakdown) or ATR-based stop
            if price > dr2 or (i > 0 and close[i-1] < dr2 and price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_1d_Pivot_R2S2_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0