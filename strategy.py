#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions with mean reversion tendency.
# In ranging markets (CHOP > 61.8), fade extremes; in trending markets (CHOP < 38.2), breakout continuation.
# Uses 1d CHOP regime filter to adapt strategy: mean reversion in chop, trend following in trends.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume > 1.5x 20-bar average confirms participation.
# Discrete position sizing at ±0.25 to limit fee drag. Target: 80-120 total trades over 4 years (20-30/year).

name = "6h_WilliamsR_1dEMA34_CHOP_Regime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA34 and CHOP regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d CHOP (Choppiness Index) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # CHOP = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = pd.Series(chop_raw).rolling(window=1, min_periods=1).values  # align to same length
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 6h Williams %R (%R = (Highest High - Close) / (Highest High - Lowest Low) * -100)
    lookback_period = 14
    highest_high = pd.Series(high).rolling(window=lookback_period, min_periods=lookback_period).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_period, min_periods=lookback_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # warmup for EMA34, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_chop = chop_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Regime-based logic
        if curr_chop > 61.8:  # Choppy/ranging market - mean reversion
            # Long: Williams %R oversold (< -80) and above 1d EMA34 (avoid strong downtrend)
            if (curr_wr < -80 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) and below 1d EMA34 (avoid strong uptrend)
            elif (curr_wr > -20 and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        else:  # Trending market (CHOP <= 61.8) - breakout continuation
            # Long: Williams %R recovering from oversold (> -80) and above 1d EMA34
            if (curr_wr > -80 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R declining from overbought (< -20) and below 1d EMA34
            elif (curr_wr < -20 and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
    
    return signals