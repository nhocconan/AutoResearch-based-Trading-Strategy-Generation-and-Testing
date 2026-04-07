#!/usr/bin/env python3
"""
12h_breakout_volume_regime_v1
Hypothesis: On 12h timeframe, enter long when price breaks above the 1-week high with above-average volume and low volatility regime (ATR percentile < 50). Enter short when price breaks below the 1-week low with above-average volume and low volatility. Exit when price returns to the 1-week midpoint or volatility increases. Uses weekly price channels for structural breaks, volume for confirmation, and ATR percentile for regime filtering. Designed to work in both bull (breakouts continue) and bear (mean reversion at extremes) markets by filtering for low volatility breakouts which have higher success rates.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_breakout_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-week data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly high, low, and midpoint
    weekly_high = high_1w
    weekly_low = low_1w
    weekly_mid = (high_1w + low_1w) / 2.0
    
    # Align to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # Calculate 1-week ATR for volatility regime filter
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile rank (52-week lookback for 1-year)
    atr_percentile = pd.Series(atr_1w).rolling(window=52, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if data not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_mid_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR percentile below 50th percentile
        low_vol = atr_percentile_aligned[i] < 0.5
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly midpoint or volatility increases significantly
            if close[i] <= weekly_mid_aligned[i] or atr_percentile_aligned[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly midpoint or volatility increases significantly
            if close[i] >= weekly_mid_aligned[i] or atr_percentile_aligned[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if low_vol and vol_ok:
                # Breakout above weekly high with volume - go long
                if close[i] > weekly_high_aligned[i] and close[i-1] <= weekly_high_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Breakout below weekly low with volume - go short
                elif close[i] < weekly_low_aligned[i] and close[i-1] >= weekly_low_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals