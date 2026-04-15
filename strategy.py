#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop for daily pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points from prior day (H+L+C)/3
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    # Daily R1: 2*P - L
    daily_r1 = 2 * daily_pivot - daily_low
    # Daily S1: 2*P - H
    daily_s1 = 2 * daily_pivot - daily_high
    # Daily R2: P + (H - L)
    daily_r2 = daily_pivot + (daily_high - daily_low)
    # Daily S2: P - (H - L)
    daily_s2 = daily_pivot - (daily_high - daily_low)
    
    # Align daily pivot levels to 6h
    daily_pivot_6h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_6h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_6h = align_htf_to_ltf(prices, df_1d, daily_s1)
    daily_r2_6h = align_htf_to_ltf(prices, df_1d, daily_r2)
    daily_s2_6h = align_htf_to_ltf(prices, df_1d, daily_s2)
    
    # Get 1w HTF data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA21 for trend direction
    weekly_close = df_1w['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_6h = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 6h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_pivot_6h[i]) or np.isnan(daily_r1_6h[i]) or 
            np.isnan(daily_s1_6h[i]) or np.isnan(daily_r2_6h[i]) or 
            np.isnan(daily_s2_6h[i]) or np.isnan(weekly_ema21_6h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above daily R1 with volume confirmation
        # 2. Weekly trend filter: price above weekly EMA21 (bullish bias)
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        if (close[i] > daily_r1_6h[i] and
            close[i] > weekly_ema21_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below daily S1 with volume confirmation
        # 2. Weekly trend filter: price below weekly EMA21 (bearish bias)
        # 3. Volatility filter: ATR > 0.3% of price
        elif (close[i] < daily_s1_6h[i] and
              close[i] < weekly_ema21_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_DailyPivot_R1S1_1w_EMA21_Volume_ATR_Filter_v1"
timeframe = "6h"
leverage = 1.0