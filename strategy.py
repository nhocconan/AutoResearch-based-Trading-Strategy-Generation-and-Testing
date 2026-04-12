#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + ADX regime filter with 12h trend confirmation
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    # ADX > 25 indicates trending market (use Elder Ray for direction)
    # ADX < 20 indicates ranging market (fade extreme Elder Ray values)
    # 12h EMA50 trend filter: only take signals aligned with higher timeframe trend
    # Discrete sizing 0.25 to minimize fee churn. Target: 20-40 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Elder Ray components (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend
        bullish_12h = close[i] > ema50_12h_aligned[i]
        bearish_12h = close[i] < ema50_12h_aligned[i]
        
        # Regime determination
        trending = adx[i] > 25
        ranging = adx[i] < 20
        
        long_entry = False
        short_entry = False
        
        if trending:
            # In trending market, use Elder Ray for direction
            # Long when Bull Power is positive and rising
            if i > 1:
                bull_power_rising = bull_power[i] > bull_power[i-1]
                bear_power_falling = bear_power[i] < bear_power[i-1]
                long_entry = bullish_12h and bull_power[i] > 0 and bull_power_rising
                short_entry = bearish_12h and bear_power[i] > 0 and bear_power_falling
        elif ranging:
            # In ranging market, fade extreme Elder Ray values
            # Long when Bear Power is extremely negative (oversold)
            # Short when Bull Power is extremely positive (overbought)
            long_entry = bear_power[i] < np.percentile(bear_power[max(0, i-50):i+1], 10) if i >= 50 else False
            short_entry = bull_power[i] > np.percentile(bull_power[max(0, i-50):i+1], 90) if i >= 50 else False
        
        # Exit logic: opposite signal or regime change to extreme
        long_exit = False
        short_exit = False
        
        if trending:
            long_exit = bearish_12h or (bull_power[i] < 0 and i > 0 and bull_power[i] < bull_power[i-1])
            short_exit = bullish_12h or (bear_power[i] < 0 and i > 0 and bear_power[i] < bear_power[i-1])
        else:  # ranging or transition
            long_exit = bear_power[i] > np.percentile(bear_power[max(0, i-50):i+1], 90) if i >= 50 else False
            short_exit = bull_power[i] < np.percentile(bull_power[max(0, i-50):i+1], 10) if i >= 50 else False
        
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

name = "6h_12h_elder_ray_adx_regime_v2"
timeframe = "6h"
leverage = 1.0