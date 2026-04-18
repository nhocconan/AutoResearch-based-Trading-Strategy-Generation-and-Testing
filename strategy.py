#!/usr/bin/env python3
"""
12h_1d_TurtleChannel_Breakout_VolumeRegime
Hypothesis: 12h timeframe with 1d context using Turtle-style Donchian breakouts (20-period) combined with 1d trend filter (EMA50) and volume confirmation. Uses choppiness regime filter to avoid ranging markets. Designed for 20-50 trades/year to minimize fee drag. Works in bull (breakouts) and bear (mean-reversion via regime filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and regime
    df_1d = get_htf_data(prices, '1d')
    
    # 12h Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d Choppiness Index (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]
    
    # ATR (14-period)
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period] = np.nansum(tr[1:atr_period+1]) if not np.isnan(tr).all() else 0
    for i in range(atr_period + 1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Sum of True Range for denominator
    sum_tr = np.zeros_like(tr)
    sum_tr[atr_period] = np.nansum(tr[1:atr_period+1]) if not np.isnan(tr).all() else 0
    for i in range(atr_period + 1, len(tr)):
        sum_tr[i] = sum_tr[i-1] + tr[i]
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Chop = 100 * log10(sum(tr) / (max(hh) - min(ll))) / log10(atr_period)
    range_hl = hh - ll
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(sum_tr / range_hl) / np.log10(atr_period)
    
    # Align 1d indicators to 12h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Precompute session filter (08-20 UTC) - optional but helps
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.3 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Regime filter: chop < 61.8 for trending, chop > 61.8 for ranging
        chop_val = chop_aligned[i]
        is_trending = chop_val < 61.8
        is_ranging = chop_val > 61.8
        
        # Only trade during active session (optional)
        in_session = session_mask[i]
        
        if position == 0:
            # Long: breakout above Donchian high + above EMA50 + volume + trending regime
            if close[i] > donchian_high[i] and price_above_ema and vol_confirm and is_trending and in_session:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + below EMA50 + volume + trending regime
            elif close[i] < donchian_low[i] and price_below_ema and vol_confirm and is_trending and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low OR trend fails (chop > 61.8) OR outside session
            if close[i] < donchian_low[i] or not is_trending or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high OR trend fails OR outside session
            if close[i] > donchian_high[i] or not is_trending or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_TurtleChannel_Breakout_VolumeRegime"
timeframe = "12h"
leverage = 1.0