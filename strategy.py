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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian to 12h
    upper_20_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_20_12h)
    lower_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_20_12h)
    
    # Get 1w HTF data for weekly context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # Get 1d HTF data for daily pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points from prior day
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Prior day's high/low/close
    prior_daily_high = np.concatenate([[daily_high[0]], daily_high[:-1]])
    prior_daily_low = np.concatenate([[daily_low[0]], daily_low[:-1]])
    prior_daily_close = np.concatenate([[daily_close[0]], daily_close[:-1]])
    
    # Daily pivot: (H+L+C)/3
    daily_pivot = (prior_daily_high + prior_daily_low + prior_daily_close) / 3.0
    # Daily R1: 2*P - L
    daily_r1 = 2 * daily_pivot - prior_daily_low
    # Daily S1: 2*P - H
    daily_s1 = 2 * daily_pivot - prior_daily_high
    
    # Align daily pivot levels to 12h
    daily_pivot_12h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_12h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_12h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: 00-24 UTC (all hours for 12h)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_12h_aligned[i]) or np.isnan(lower_20_12h_aligned[i]) or 
            np.isnan(weekly_ema21_aligned[i]) or np.isnan(daily_pivot_12h[i]) or 
            np.isnan(daily_r1_12h[i]) or np.isnan(daily_s1_12h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 12h price breaks above 12h Donchian upper (20) - bullish breakout
        # 2. Weekly EMA21 filter: price above weekly EMA21 (bullish weekly trend)
        # 3. Price above daily pivot (bullish bias from prior day)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_12h_aligned[i] and
            close[i] > weekly_ema21_aligned[i] and
            close[i] > daily_pivot_12h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h price breaks below 12h Donchian lower (20) - bearish breakdown
        # 2. Weekly EMA21 filter: price below weekly EMA21 (bearish weekly trend)
        # 3. Price below daily pivot (bearish bias from prior day)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_12h_aligned[i] and
              close[i] < weekly_ema21_aligned[i] and
              close[i] < daily_pivot_12h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_12h_Donchian20_1w_EMA21_1d_Pivot_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0