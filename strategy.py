#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly pivot bias and volume confirmation
# Uses 1d primary timeframe with 1w HTF for trend filter. Designed to work in both bull and bear
# by requiring strong breakouts with volume, avoiding choppy markets. Target: 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Donchian and pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 1d (same timeframe, no alignment needed but using for consistency)
    upper_20_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # Weekly high/low/close from 5 trading days ago (prior week)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(5).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(5).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot: (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 1d
    weekly_pivot_1d = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_1d = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_1d = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 1d - always true)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 1d, kept for structure
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_1d_aligned[i]) or np.isnan(lower_20_1d_aligned[i]) or 
            np.isnan(weekly_pivot_1d[i]) or np.isnan(weekly_r1_1d[i]) or 
            np.isnan(weekly_s1_1d[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1d price breaks above 1d Donchian upper (20) - bullish breakout
        # 2. Price above weekly pivot (bullish bias from prior week)
        # 3. 1w EMA(50) uptrend: price above 1w EMA50
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_1d_aligned[i] and
            close[i] > weekly_pivot_1d[i] and
            close[i] > ema_50_1w_aligned[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 1d price breaks below 1d Donchian lower (20) - bearish breakdown
        # 2. Price below weekly pivot (bearish bias from prior week)
        # 3. 1w EMA(50) downtrend: price below 1w EMA50
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_1d_aligned[i] and
              close[i] < weekly_pivot_1d[i] and
              close[i] < ema_50_1w_aligned[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1w_EMA50_WeeklyPivot_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0