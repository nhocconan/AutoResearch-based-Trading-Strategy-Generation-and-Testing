#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R1/S1) breakout with volume confirmation and chop regime filter.
# Uses 1d for pivot calculation (prior week) and 1w for chop regime (EWCO). 
# Designed for low trade frequency (~15-25/year) to avoid fee drag, works in bull/bear via regime adaptation.
# Long: break above R1 with volume > 1.5x avg and chop < 61.8 (trending). 
# Short: break below S1 with volume > 1.5x avg and chop < 61.8.
# Exit: reverse signal or chop > 61.8 (range) to reduce whipsaw.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior complete week (5 trading days)
    # Using 1d data: weekly high/low/close from 5 days ago (shifted 5)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(5).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(5).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot: (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 12h (already aligned via index in get_htf_data, but ensure)
    weekly_pivot_12h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_12h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Get 1w HTF data for chop regime (EWCO - Ehlers' Choppy Index)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Ehlers Choppy Index (EWCO) on 1w: 100 * (sum of ATR1 over n) / (sum of true range over n)
    # Simplified: using ATR(14) and max-min over 14 periods
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Max/min over 14 periods for denominator
    max_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_1w = max_high_1w - min_low_1w
    
    # EWCO = 100 * (sum of ATR1 over 14) / (sum of true range over 14) 
    # Approximate: sum of ATR1 ≈ atr_1w * 14, sum of true range ≈ range_1w * 14 (crude but functional)
    # Better: use rolling sum of ATR and rolling sum of (high-low) but we approximate for speed
    sum_atr_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    sum_tr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    ewco = 100 * (sum_atr_1w / (sum_tr_1w + 1e-10))
    
    # Align EWCO to 12h
    ewco_12h = align_htf_to_ltf(prices, df_1w, ewco)
    
    # Calculate 12h ATR(14) for volatility filter (optional, but keep for robustness)
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_12h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume ratio: current vs 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 12h - always true, kept for structure)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_12h[i]) or np.isnan(weekly_s1_12h[i]) or 
            np.isnan(ewco_12h[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(atr_14_12h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when chop < 61.8 (trending market)
        is_trending = ewco_12h[i] < 61.8
        
        # Long conditions:
        # 1. 12h price breaks above weekly R1
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Trending regime (chop < 61.8)
        if (close[i] > weekly_r1_12h[i] and
            volume_ratio[i] > 1.5 and
            is_trending):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h price breaks below weekly S1
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Trending regime (chop < 61.8)
        elif (close[i] < weekly_s1_12h[i] and
              volume_ratio[i] > 1.5 and
              is_trending):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_WeeklyPivot_R1S1_1w_EWCO_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0