#!/usr/bin/env python3
"""
6h_ADX_Regime_ElderRay_VolumeConfirm
Hypothesis: On 6h timeframe, Elder Ray (Bull/Bear Power) signals filtered by ADX regime and volume confirmation work in both bull and bear markets.
- ADX > 25 indicates trending market: use Elder Ray for trend-following entries
- ADX < 20 indicates ranging market: use Elder Ray for mean-reversion exits
- Volume spike confirms institutional participation
- Weekly trend filter from 1d EMA50 ensures alignment with intermediate trend
- Target: 12-30 trades/year (50-150 over 4 years) with discrete position sizing
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_wilder_atr(high, low, close, period):
    """Calculate Wilder's ATR (used in ADX) with min_periods"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = np.zeros_like(tr)
    atr[:period-1] = np.nan
    atr[period-1] = np.nanmean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) with min_periods"""
    if len(high) < period * 2:
        return np.full_like(high, np.nan)
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Calculate ATR
    atr = calculate_wilder_atr(high, low, close, period)
    # Calculate +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter and ADX regime (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ADX for regime detection
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (50) + volume MA (20) + ADX (14*2)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Elder Ray components
        bull_power = curr_high - ema_50_1d_aligned[i]
        bear_power = ema_50_1d_aligned[i] - curr_low
        
        if position == 0:
            # Look for entry signals
            # Trending regime (ADX > 25): Elder Ray continuation
            if adx_1d_aligned[i] > 25:
                long_entry = (bull_power > 0 and volume_spike[i] and curr_close > ema_50_1d_aligned[i])
                short_entry = (bear_power > 0 and volume_spike[i] and curr_close < ema_50_1d_aligned[i])
            # Ranging regime (ADX < 20): Elder Ray mean reversion at extremes
            elif adx_1d_aligned[i] < 20:
                # In range, look for reversals from extreme levels
                long_entry = (bear_power > 0 and volume_spike[i] and curr_close < ema_50_1d_aligned[i] * 0.995)
                short_entry = (bull_power > 0 and volume_spike[i] and curr_close > ema_50_1d_aligned[i] * 1.005)
            else:
                # Transition regime: no entries
                long_entry = False
                short_entry = False
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if trend weakens (ADX < 20) or Elder Ray turns negative
            if adx_1d_aligned[i] < 20 or bull_power < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if trend weakens (ADX < 20) or Elder Ray turns negative
            if adx_1d_aligned[i] < 20 or bear_power < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Regime_ElderRay_VolumeConfirm"
timeframe = "6h"
leverage = 1.0