#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions.
# Williams %R < -80 = oversold (buy signal in uptrend)
# Williams %R > -20 = overbought (sell signal in downtrend)
# Strategy: In trending markets (ADX > 25), buy oversold dips in uptrend, sell overbought rallies in downtrend.
# In ranging markets (ADX < 20), mean revert at Williams %R extremes.
# Volume spike confirms institutional participation.
# Designed for ~20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ADX calculation for regime filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM-
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    tr_smooth = np.where(tr_smooth == 0, 1e-10, tr_smooth)
    
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        if adx[i] > 25:  # Trending market
            # Buy oversold dips in uptrend, sell overbought rallies in downtrend
            if close[i] > ema34_1d_aligned[i] and williams_r[i] < -80 and volume_filter[i]:  # Uptrend + oversold
                signals[i] = 0.25
                position = 1
            elif close[i] < ema34_1d_aligned[i] and williams_r[i] > -20 and volume_filter[i]:  # Downtrend + overbought
                signals[i] = -0.25
                position = -1
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:  # Ranging market (ADX < 25) or neutral
            # Mean revert at Williams %R extremes
            if williams_r[i] < -80 and volume_filter[i]:  # Oversold
                signals[i] = 0.25
                position = 1
            elif williams_r[i] > -20 and volume_filter[i]:  # Overbought
                signals[i] = -0.25
                position = -1
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_1dEMA34_ADX_VolumeFilter"
timeframe = "12h"
leverage = 1.0