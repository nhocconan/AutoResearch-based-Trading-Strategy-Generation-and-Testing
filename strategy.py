#!/usr/bin/env python3
"""
6h_WeeklyPivot_R2S2_Breakout_Volume_TrendFilter
Hypothesis: Weekly pivot R2/S2 levels act as strong support/resistance on 6h timeframe. Breakouts with volume confirmation and trend filter (price above/below weekly EMA20) capture directional moves. Works in both bull/bear markets by using weekly context and volatility filter to avoid chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and EMA (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using standard formula
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    pivot = (high_1w + low_1w + close_1w) / 3
    r2 = pivot + (high_1w - low_1w)  # R2 = pivot + (high - low)
    s2 = pivot - (high_1w - low_1w)  # S2 = pivot - (high - low)
    
    # Shift by 1 to use previous week's levels only
    r2_prev = r2.shift(1).values
    s2_prev = s2.shift(1).values
    
    # Align to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_prev)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_prev)
    
    # Get weekly data for EMA trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # ATR for volatility filter (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 20-period average (avoid chop)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_ma
    
    # Volume spike: 2.0x 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    bars_since_entry = 0  # track holding period
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: break above R2 with volume spike, price above weekly EMA, and sufficient volatility
            if price > r2_val and vol_spike and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with volume spike, price below weekly EMA, and sufficient volatility
            elif price < s2_val and vol_spike and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Minimum holding period: 3 bars (18 hours for 6h)
            if bars_since_entry < 3:
                signals[i] = 0.25
                bars_since_entry += 1
            else:
                signals[i] = 0.25
                # Exit: price returns to S2 or breaks below weekly EMA
                if price <= s2_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            # Minimum holding period: 3 bars (18 hours for 6h)
            if bars_since_entry < 3:
                signals[i] = -0.25
                bars_since_entry += 1
            else:
                signals[i] = -0.25
                # Exit: price returns to R2 or breaks above weekly EMA
                if price >= r2_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "6h_WeeklyPivot_R2S2_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0