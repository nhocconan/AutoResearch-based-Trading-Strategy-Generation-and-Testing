# 6h ATR Breakout with Weekly Trend Filter and Volume Confirmation
# Uses ATR breakout from 15-period high/low for entry, weekly ADX for trend strength,
# and volume spike for confirmation. Designed for 12-37 trades/year.
# Works in both bull and breakouts: ATR breakouts capture volatility expansion,
# while weekly ADX filter ensures we only trade in strong trends, avoiding whipsaws in ranging markets.
name = "6h_ATRBreakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly ADX for trend strength
    # Calculate True Range
    tr1 = df_weekly['high'] - df_weekly['low']
    tr2 = abs(df_weekly['high'] - df_weekly['close'].shift(1))
    tr3 = abs(df_weekly['low'] - df_weekly['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = df_weekly['high'] - df_weekly['high'].shift(1)
    dm_minus = df_weekly['low'].shift(1) - df_weekly['low']
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smooth TR and DM
    tr_14 = tr.rolling(window=14, min_periods=14).sum()
    dm_plus_14 = dm_plus.rolling(window=14, min_periods=14).sum()
    dm_minus_14 = dm_minus.rolling(window=14, min_periods=14).sum()
    
    # Calculate DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # Calculate ADX
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_weekly = adx.values
    
    # Align weekly ADX to 6h
    adx_6h = align_htf_to_ltf(prices, df_weekly, adx_weekly)
    
    # 15-period ATR for breakout levels (on 6h data)
    tr_6h = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            abs(high[1:] - close[:-1]),
            abs(low[1:] - close[:-1])
        )
    )
    tr_6h = np.concatenate([[np.nan], tr_6h])  # align with index
    atr_15 = pd.Series(tr_6h).rolling(window=15, min_periods=15).mean().values
    
    # 15-period high and low for breakout levels
    high_15 = pd.Series(high).rolling(window=15, min_periods=15).max().values
    low_15 = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15, 20)  # ensure we have enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_6h[i]) or np.isnan(atr_15[i]) or 
            np.isnan(high_15[i]) or np.isnan(low_15[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout levels: ATR multiple of 1.0 added/subtracted from 15-period high/low
        breakout_up = high_15[i] + atr_15[i]
        breakout_down = low_15[i] - atr_15[i]
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        # Trend filter: weekly ADX > 25 indicates strong trend
        strong_trend = adx_6h[i] > 25
        
        if position == 0:
            # Long: Price breaks above breakout_up with strong trend and volume spike
            if close[i] > breakout_up and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below breakout_down with strong trend and volume spike
            elif close[i] < breakout_down and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below 15-period low OR trend weakens
            if close[i] < low_15[i] or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above 15-period high OR trend weakens
            if close[i] > high_15[i] or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf