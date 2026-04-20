#!/usr/bin/env python3
# 1d_Weekly_Volume_Regime_Strategy
# Hypothesis: Weekly trend + volume confirmation + daily regime filter (chop) captures major moves while avoiding whipsaws.
# In bull markets: go long when price above weekly EMA20 with volume surge and low chop.
# In bear markets: go short when price below weekly EMA20 with volume surge and low chop.
# Uses weekly EMA for trend, volume spike for confirmation, daily chop filter to avoid ranging markets.
# Target: 15-25 trades/year to minimize fee drag.

name = "1d_Weekly_Volume_Regime_Strategy"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend
    weekly_close = df_weekly['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema20)
    
    # Daily chop filter (avoid ranging markets)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_max_min = max_high - min_low
    range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
    chop = 100 * np.log10(atr.sum() / range_max_min) / np.log10(14)
    chop = pd.Series(chop).ewm(span=14, adjust=False, min_periods=14).mean().values
    chop_filter = chop < 61.8  # Trending market when chop < 61.8
    
    # Volume confirmation: volume > 2.0x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema20_aligned[i]) or np.isnan(chop_filter[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly EMA20 + volume surge + trending market
            if close[i] > weekly_ema20_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA20 + volume surge + trending market
            elif close[i] < weekly_ema20_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below weekly EMA20 or chop increases (ranging)
            if close[i] < weekly_ema20_aligned[i] or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above weekly EMA20 or chop increases (ranging)
            if close[i] > weekly_ema20_aligned[i] or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals