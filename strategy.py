#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d EMA34 trend + volume spike
# Uses Choppiness Index (CHOP) to detect ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In ranging markets: mean reversion at Bollinger Bands (20,2)
# In trending markets: trend following with EMA crossover
# Volume spike confirms momentum. Designed for low trade frequency in both bull and bear.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "12h_ChopRegime_EMA_BB"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data once
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Weekly EMA(20) for higher timeframe trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Choppiness Index (14-period) on 12h data
    def true_range(h, l, pc):
        tr1 = h - l
        tr2 = np.abs(h - pc)
        tr3 = np.abs(l - pc)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_max_min = highest_high - lowest_low
    chop = np.full_like(close, 50.0, dtype=float)
    mask = range_max_min != 0
    chop[mask] = 100 * np.log10(atr14[mask] * 14 / range_max_min[mask]) / np.log10(14)
    
    # Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    
    # EMA crossover (fast=9, slow=21)
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike: current > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(sma20[i]) or
            np.isnan(std20[i]) or np.isnan(ema9[i]) or np.isnan(ema21[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        ema34_1d_val = ema34_1d_aligned[i]
        ema20_1w_val = ema20_1w_aligned[i]
        sma20_val = sma20[i]
        std20_val = std20[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        ema9_val = ema9[i]
        ema21_val = ema21[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Ranging market: CHOP > 61.8 -> mean reversion at Bollinger Bands
            if chop_val > 61.8:
                # Long at lower BB with uptrend bias
                if close[i] <= bb_lower_val and close[i] > ema34_1d_val and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short at upper BB with downtrend bias
                elif close[i] >= bb_upper_val and close[i] < ema34_1d_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Trending market: CHOP < 38.2 -> follow EMA crossover with higher timeframe filter
            elif chop_val < 38.2:
                # Long: EMA9 > EMA21 and above weekly EMA
                if ema9_val > ema21_val and close[i] > ema20_1w_val and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: EMA9 < EMA21 and below weekly EMA
                elif ema9_val < ema21_val and close[i] < ema20_1w_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: range breaks down OR trend reverses
            if chop_val > 61.8 and close[i] >= sma20_val:  # return to mean in ranging
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and ema9_val < ema21_val:  # trend reversed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: range breaks up OR trend reverses
            if chop_val > 61.8 and close[i] <= sma20_val:  # return to mean in ranging
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and ema9_val > ema21_val:  # trend reversed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals