#!/usr/bin/env python3
"""
6h_12h_WilR_Regime_Adaptive_v1
Hypothesis: Use Williams %R on 12h with regime detection (trending vs ranging) to adapt strategy.
- In trending regimes (ADX > 25): trade breakouts of WilR oversold/overbought levels
- In ranging regimes (ADX <= 25): mean-revert at WilR extremes
- Uses volume confirmation to avoid false signals
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drift.
Works in bull via trend-following breakouts, in bear via mean-reversion from extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_WilR_Regime_Adaptive_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Williams %R and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Williams %R (14-period)
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    close_12h = df_12h['close'].values
    willr = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)  # avoid division by zero
    
    # ADX (14-period) for regime detection
    plus_dm = pd.Series(df_12h['high']).diff()
    minus_dm = pd.Series(df_12h['low']).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = pd.Series(df_12h['high']) - pd.Series(df_12h['low'])
    tr2 = abs(pd.Series(df_12h['high']) - pd.Series(df_12h['close']).shift())
    tr3 = abs(pd.Series(df_12h['low']) - pd.Series(df_12h['close']).shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align indicators to 6h timeframe
    willr_aligned = align_htf_to_ltf(prices, df_12h, willr)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx.values)
    
    # Volume average (20-period) on 6s
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(willr_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Regime detection
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] <= 25
        
        # Williams %R levels
        oversold = willr_aligned[i] < -80
        overbought = willr_aligned[i] > -20
        
        if trending:
            # Trend-following: buy oversold, sell overbought
            long_entry = oversold and vol_spike
            short_entry = overbought and vol_spike
            # Exit when Williams %R returns to neutral zone
            long_exit = willr_aligned[i] > -50
            short_exit = willr_aligned[i] < -50
        else:
            # Mean-reversion: fade extremes
            long_entry = oversold and vol_spike
            short_entry = overbought and vol_spike
            # Exit when Williams %R returns to midpoint
            long_exit = willr_aligned[i] > -50
            short_exit = willr_aligned[i] < -50
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals