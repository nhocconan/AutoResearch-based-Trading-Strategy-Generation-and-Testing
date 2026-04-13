#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d regime filter.
    # Elder Ray measures bull/bear power via EMA(13): Bull Power = High - EMA, Bear Power = EMA - Low.
    # Regime filter: 1d ADX > 25 for trending, ADX < 20 for ranging.
    # In trending markets: trade Elder Ray extremes (strong Bull/Bear Power breakouts).
    # In ranging markets: fade Elder Ray extremes (mean reversion at extremes).
    # Target: 50-150 total trades over 4 years = 12-37/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_period = 13
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Elder Ray components
    bull_power = high - ema  # Bull Power = High - EMA
    bear_power = ema - low   # Bear Power = EMA - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema[i])):
            signals[i] = 0.0
            continue
        
        # Regime detection
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        # Elder Ray extremes (using 20-period lookback for normalization)
        lookback = 20
        if i >= lookback:
            bull_ma = np.nanmean(bull_power[i-lookback:i])
            bull_std = np.nanstd(bull_power[i-lookback:i])
            bear_ma = np.nanmean(bear_power[i-lookback:i])
            bear_std = np.nanstd(bear_power[i-lookback:i])
            
            # Normalized Elder Ray (z-score)
            bull_z = (bull_power[i] - bull_ma) / (bull_std + 1e-8)
            bear_z = (bear_power[i] - bear_ma) / (bear_std + 1e-8)
            
            # Entry thresholds
            extreme_threshold = 1.5
            
            if is_trending:
                # Trending: trade in direction of Elder Ray extremes
                long_entry = bull_z > extreme_threshold
                short_entry = bear_z > extreme_threshold
                # Exit when Elder Ray returns to mean
                long_exit = bull_z < 0.5
                short_exit = bear_z < 0.5
            elif is_ranging:
                # Ranging: fade Elder Ray extremes (mean reversion)
                long_entry = bear_z > extreme_threshold  # Sell pressure exhaustion -> long
                short_entry = bull_z > extreme_threshold  # Buy pressure exhaustion -> short
                # Exit when Elder Ray returns to mean
                long_exit = bear_z < 0.5
                short_exit = bull_z < 0.5
            else:
                # Transition regime: no trades
                long_entry = short_entry = long_exit = short_exit = False
        else:
            # Not enough data for normalization
            long_entry = short_entry = long_exit = short_exit = False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0