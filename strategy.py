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
    
    # Get daily data for ATR and ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range components
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr1[0] = high_d[0] - low_d[0]  # First bar has no previous close
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR(50) for longer-term volatility context
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR(14) / ATR(50) > 1.2 indicates high volatility
    vol_regime = np.zeros_like(atr_14)
    valid_idx = (~np.isnan(atr_14)) & (~np.isnan(atr_50)) & (atr_50 > 0)
    vol_regime[valid_idx] = (atr_14[valid_idx] / atr_50[valid_idx]) > 1.2
    
    # Align ATR-based volatility regime to 4h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    # Get weekly data for trend filter using EMA crossover
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(9) and EMA(21) for trend
    close_1w = df_1w['close'].values
    ema9_1w = pd.Series(close_1w).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Weekly trend: EMA9 > EMA21 = uptrend, EMA9 < EMA21 = downtrend
    weekly_uptrend = ema9_1w > ema21_1w
    weekly_downtrend = ema9_1w < ema21_1w
    
    # Align weekly trend to 4h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # 4h Donchian channel breakout (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, lookback)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_regime_aligned[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in high volatility regimes (ATR(14)/ATR(50) > 1.2)
        if vol_regime_aligned[i] <= 0.5:  # Not in high volatility regime
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume_filter[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Entry conditions: breakout in direction of weekly trend with volume
        long_entry = long_breakout and vol_filter and weekly_uptrend_aligned[i] > 0.5
        short_entry = short_breakout and vol_filter and weekly_downtrend_aligned[i] > 0.5
        
        # Exit conditions: opposite breakout level
        long_exit = close[i] < lowest_low[i] and position == 1
        short_exit = close[i] > highest_high[i] and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_VolRegime_Breakout_WeeklyEMA_Trend_Volume_Session"
timeframe = "4h"
leverage = 1.0