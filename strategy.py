#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1/S1) for mean reversion entries in the direction of 1w EMA200 trend.
# In 1w uptrend (close > EMA200), go long when price touches or breaks below S1 with volume spike.
# In 1w downtrend (close < EMA200), go short when price touches or breaks above R1 with volume spike.
# Uses choppiness regime filter (CHOP > 61.8) to avoid whipsaw in strong trends.
# Designed for low trade frequency (20-40/year) to minimize fee drag while capturing mean reversion in trending markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = pivot + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pivot - (high_1d - low_1d) * 1.1 / 12.0
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1w Indicators: EMA200 for trend direction ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 1d Indicators: Choppiness Index (CHOP) for regime filter ===
    atr_1d = pd.Series(np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - close_1d.shift(1))), np.abs(low_1d - close_1d.shift(1)))).rolling(window=14, min_periods=14).mean().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(max_high_1d - min_low_1d) * np.sqrt(14)
    chop_1d = 100 * np.log10(atr_1d * np.sqrt(14) / chop_denom) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in choppy markets (CHOP > 61.8)
        if chop_1d_aligned[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 1w uptrend (close > EMA200)
        # 2. Price touches or breaks below 1d S1 level
        # 3. Volume confirmation
        if (close[i] > ema_200_1w_aligned[i]) and (close[i] <= s1_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1w downtrend (close < EMA200)
        # 2. Price touches or breaks above 1d R1 level
        # 3. Volume confirmation
        elif (close[i] < ema_200_1w_aligned[i]) and (close[i] >= r1_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_1d_Camarilla_R1S1_1wEMA200_Volume_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0