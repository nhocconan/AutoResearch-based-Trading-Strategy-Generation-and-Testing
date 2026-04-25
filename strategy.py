#!/usr/bin/env python3
"""
4h_Williams_VIX_Fix_MeanReversion_1dTrendFilter
Hypothesis: 4-hour Williams VIX Fix mean reversion with 1-day trend filter and volume confirmation.
Long when VIX Fix < 20 (extreme fear) in 1-day uptrend with volume confirmation (>1.5x 20-period average).
Short when VIX Fix > 80 (extreme greed) in 1-day downtrend with volume confirmation.
Exit via opposite VIX Fix threshold (40 for long exit, 60 for short exit) or ATR trailing stop (1.5*ATR).
Williams VIX Fix measures market fear/greed - effective in both bull and bear markets for mean reversion trades.
Volume confirmation ensures mean reversion has conviction. 1-day trend filter aligns with higher timeframe bias.
Designed for ~25-60 trades over 4 years (6-15/year) via tight VIX Fix extreme conditions.
"""

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
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for EMA and VIX Fix calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams VIX Fix on 1d data
    # VIX Fix = ( (Highest High in 22 periods - Low) / (Highest High in 22 periods - Lowest Low in 22 periods) ) * 100
    lookback = 22
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    
    vix_fix = ((highest_high - low_1d) / hh_ll) * 100
    vix_fix_aligned = align_htf_to_ltf(prices, df_1d, vix_fix)
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, lookback)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vix_fix_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        vix = vix_fix_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: extreme fear (VIX Fix < 20) with volume confirmation
                long_signal = (vix < 20) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: extreme greed (VIX Fix > 80) with volume confirmation
                short_signal = (vix > 80) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (1.5*ATR from extreme)
            atr_stop = long_extreme - 1.5 * atr[i]
            # 2. VIX Fix exits extreme fear zone (VIX Fix > 40)
            if close[i] <= atr_stop or vix > 40:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (1.5*ATR from extreme)
            atr_stop = short_extreme + 1.5 * atr[i]
            # 2. VIX Fix exits extreme greed zone (VIX Fix < 60)
            if close[i] >= atr_stop or vix < 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Williams_VIX_Fix_MeanReversion_1dTrendFilter"
timeframe = "4h"
leverage = 1.0