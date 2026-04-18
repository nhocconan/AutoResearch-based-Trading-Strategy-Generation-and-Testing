# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_S1R1_Breakout_VolumeSpike_1wTrend_v1
Hypothesis: Weekly trend direction filtered by EMA(50) combined with daily Camarilla S1/R1 breakouts on 12h timeframe provides high-probability entries in both bull and bear markets. Weekly trend filter reduces counter-trend trades, volume surge confirms breakout strength, and Camarilla levels provide institutional-grade support/resistance. Target: 20-30 trades/year on 12h timeframe.
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
    
    # Get daily data for Camarilla pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Camarilla formula: range = (high - low)
    # S1 = close - (range * 1.1 / 12)
    # R1 = close + (range * 1.1 / 12)
    range_1d = high_1d - low_1d
    s1 = close_1d - (range_1d * 1.1 / 12)
    r1 = close_1d + (range_1d * 1.1 / 12)
    
    # Shift by 1 to use previous day's levels only
    s1_prev = s1.shift(1).values
    r1_prev = r1.shift(1).values
    
    # Align to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close']
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for volatility filter (14-period on 12h)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 20-period average (avoid chop)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_ma
    
    # Volume spike: 2.5x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    bars_since_entry = 0  # track holding period
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: break above R1 with volume spike, price above weekly EMA, and sufficient volatility
            if price > r1_val and vol_spike and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike, price below weekly EMA, and sufficient volatility
            elif price < s1_val and vol_spike and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Minimum holding period: 2 bars (24 hours for 12h)
            if bars_since_entry < 2:
                signals[i] = 0.25
                bars_since_entry += 1
            else:
                signals[i] = 0.25
                # Exit: price returns to S1 or breaks below weekly EMA
                if price <= s1_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            # Minimum holding period: 2 bars (24 hours for 12h)
            if bars_since_entry < 2:
                signals[i] = -0.25
                bars_since_entry += 1
            else:
                signals[i] = -0.25
                # Exit: price returns to R1 or breaks above weekly EMA
                if price >= r1_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "12h_Camarilla_Pivot_S1R1_Breakout_VolumeSpike_1wTrend_v1"
timeframe = "12h"
leverage = 1.0