#!/usr/bin/env python3
"""
12h_1d_Pivot_R2S2_Breakout_Volume_ATRFilter
Hypothesis: Daily Pivot Point R2/S2 levels act as key support/resistance in 12h timeframe. Breakout above R2 or below S2 with volume confirmation and ATR filter captures strong momentum moves. Works in both bull and bear markets by using volatility filter to avoid false breakouts in ranging conditions. Target: 12-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Pivot Point calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily Pivot Point levels
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    P = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    r2_daily = P + range_daily
    s2_daily = P - range_daily
    
    # Align daily Pivot levels to 12h timeframe
    r2_daily_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    s2_daily_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    # ATR(14) for volatility filter
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros_like(close)
    for i in range(len(atr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    # Volatility filter: ATR > 20-period average ATR (avoid low volatility periods)
    atr_avg = np.zeros_like(atr)
    for i in range(len(atr)):
        if i >= 20:
            atr_avg[i] = np.mean(atr[i-20:i])
        else:
            atr_avg[i] = np.mean(atr[:i+1]) if i > 0 else atr[i]
    volatility_filter = atr > (0.8 * atr_avg)  # Only trade when volatility is above 80% of average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(r2_daily_aligned[i]) or np.isnan(s2_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r2 = r2_daily_aligned[i]
        s2 = s2_daily_aligned[i]
        vol_ok = volume_filter[i]
        vol_filter_ok = volatility_filter[i]
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume and volatility confirmation
            if price > r2 and vol_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S2 with volume and volatility confirmation
            elif price < s2 and vol_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Pivot Point or breaks below S1 (failed breakout)
            P_daily = (high_daily + low_daily + close_daily) / 3.0
            P_aligned = align_htf_to_ltf(prices, df_daily, P_daily)
            H_daily = high_daily
            L_daily = low_daily
            s1_daily = 2 * P_daily - H_daily
            s1_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
            
            if not np.isnan(P_aligned[i]) and not np.isnan(s1_aligned[i]):
                if price < P_aligned[i] or price < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Pivot Point or breaks above R1 (failed breakdown)
            P_daily = (high_daily + low_daily + close_daily) / 3.0
            P_aligned = align_htf_to_ltf(prices, df_daily, P_daily)
            H_daily = high_daily
            L_daily = low_daily
            r1_daily = 2 * P_daily - L_daily
            r1_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
            
            if not np.isnan(P_aligned[i]) and not np.isnan(r1_aligned[i]):
                if price > P_aligned[i] or price > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R2S2_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0