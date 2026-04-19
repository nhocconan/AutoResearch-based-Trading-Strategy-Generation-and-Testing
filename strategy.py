#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation.
# In chop regime (CHOP > 61.8): mean reversion at Donchian edges.
# In trend regime (CHOP < 38.2): breakout continuation.
# Uses weekly trend filter to avoid counter-trend trades in strong trends.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_ChopRegime_Donchian20_Volume_WeeklyTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter (needs confirmation delay)
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Choppiness Index (14-period)
    atr_1d = np.zeros(len(high_1d))
    atr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                 abs(high_1d[i] - close_1d[i-1]), 
                 abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    
    # Align Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channel (20-period) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 14)  # Donchian and Chop need initialization
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        chop_val = chop_aligned[i]
        weekly_ema = ema34_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # Regime detection
        is_chop = chop_val > 61.8
        is_trend = chop_val < 38.2
        
        if position == 0:
            # Enter long conditions
            long_breakout = price > upper and volume_confirmed
            long_pullback = price < upper and price > lower and is_chop and volume_confirmed
            
            # Enter short conditions
            short_breakout = price < lower and volume_confirmed
            short_pullback = price > lower and price < upper and is_chop and volume_confirmed
            
            # Weekly trend filter: only take trades in direction of weekly trend
            if weekly_ema > 0:  # weekly uptrend
                if long_breakout or (long_pullback and price > weekly_ema):
                    signals[i] = 0.25
                    position = 1
            else:  # weekly downtrend
                if short_breakout or (short_pullback and price < weekly_ema):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: Donchian break below or chop extreme
            if price < lower or chop_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian break above or chop extreme
            if price > upper or chop_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals