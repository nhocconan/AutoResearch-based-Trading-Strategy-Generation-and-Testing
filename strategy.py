#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Choppiness_Filtered_Camarilla_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla pivot calculation
    daily_high = df_d['high'].values
    daily_low = df_d['low'].values
    daily_close = df_d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    pivot = (daily_high + daily_low + daily_close) / 3
    r1 = pivot + (daily_high - daily_low) * 1.1 / 12
    s1 = pivot - (daily_high - daily_low) * 1.1 / 12
    
    # Align to 12h timeframe (Camarilla levels from previous day)
    r1_aligned = align_htf_to_ltf(prices, df_d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1)
    
    # Daily EMA(34) for trend filter
    close_d = pd.Series(daily_close)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Choppiness Index on 1d timeframe (trend/range filter)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = np.maximum(high[1:], daily_close[:-1]) - np.minimum(low[1:], daily_close[:-1])
    tr1 = np.concatenate([[np.nan], tr1])
    atr1 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high1 = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    min_low1 = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    chop1 = 100 * np.log10(sum_atr1 / (max_high1 - min_low1)) / np.log10(14)
    chop1 = np.where((max_high1 - min_low1) == 0, 50, chop1)  # avoid div by zero
    chop1_aligned = align_htf_to_ltf(prices, df_d, chop1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_d_aligned[i]) or np.isnan(vol_ma20[i]) or 
            np.isnan(chop1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        # Range market: CHOP > 61.8, Trending market: CHOP < 38.2
        ranging = chop1_aligned[i] > 61.8
        
        if position == 0:
            # Long: Price breaks above R1 with volume, in ranging market, and above daily EMA trend
            if close[i] > r1_aligned[i] and vol_ok and ranging and close[i] > ema34_d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume, in ranging market, and below daily EMA trend
            elif close[i] < s1_aligned[i] and vol_ok and ranging and close[i] < ema34_d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below S1 (reversion to mean) or strong trend (CHOP < 38.2)
            if close[i] < s1_aligned[i] or chop1_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above R1 (reversion to mean) or strong trend (CHOP < 38.2)
            if close[i] > r1_aligned[i] or chop1_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals