#!/usr/bin/env python3
"""
1D Weekly EMA34 Trend with Volume Filter
Hypothesis: In trending markets (both bull and bear), price tends to respect the weekly EMA34 as dynamic support/resistance.
During uptrends, price pulls back to EMA34 and bounces; during downtrends, price rallies to EMA34 and rejects.
Volume confirmation filters out weak moves. Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
Target: 15-25 trades per year by requiring weekly EMA34 alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for EMA34 trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    # Calculate weekly EMA34
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to daily timeframe
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema34_weekly_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34 = ema34_weekly_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above weekly EMA34 with volume confirmation
            if price > ema34 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA34 with volume confirmation
            elif price < ema34 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA34
            if price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA34
            if price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1D_WeeklyEMA34_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0