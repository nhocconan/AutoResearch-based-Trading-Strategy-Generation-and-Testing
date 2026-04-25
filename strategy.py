#!/usr/bin/env python3
"""
4h Williams Alligator Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trendless periods; breakout above/below Alligator during low volatility (chop > 61.8) with volume spike and daily EMA50 trend alignment captures explosive moves after consolidation. Works in bull/bear by trading breakouts in direction of higher timeframe trend. Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Williams Alligator: SMAs of median price shifted into future"""
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().shift(3).values
    return jaw, teeth, lips

def calculate_chop(high, low, close, period):
    """Calculate Choppiness Index"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    log_sum_tr = np.log10(np.where(sum_tr <= 0, 1e-10, sum_tr))
    log_period = np.log10(period)
    chop = 100 * (log_sum_tr / log_period)
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA50 trend and Chop filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily Chop filter (chop > 61.8 = ranging market, good for breakout after consolidation)
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Williams Alligator on 4h (median price: (high+low)/2)
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss (10-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Alligator (jaw=13+8=21), EMA, volume MA, ATR, Chop
    start_idx = max(21, 50, 20, 10, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator lines: jaw (slow), teeth (medium), lips (fast)
        # When lips > teeth > jaw: uptrend; lips < teeth < jaw: downtrend
        # We trade breakouts when Alligator is sleeping (all lines intertwined)
        alligator_sleeping = (
            (abs(jaw[i] - teeth[i]) < (atr[i] * 0.5)) and
            (abs(teeth[i] - lips[i]) < (atr[i] * 0.5)) and
            (abs(lips[i] - jaw[i]) < (atr[i] * 0.5))
        )
        
        # Chop regime: we want chop > 61.8 (strong ranging/consolidation) before breakout
        chop_regime = chop_1d_aligned[i] > 61.8
        
        # Breakout conditions: price breaks above/below Alligator jaws
        breakout_long = curr_close > jaw[i]
        breakout_short = curr_close < jaw[i]
        
        if position == 0:
            # Look for entry signals - require: Alligator sleeping + chop regime + volume spike + daily EMA50 trend alignment
            long_entry = breakout_long and alligator_sleeping and chop_regime and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = breakout_short and alligator_sleeping and chop_regime and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Alligator teeth or trend change
            if curr_close < teeth[i] or curr_close < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Alligator teeth or trend change
            if curr_close > teeth[i] or curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_Breakout_1dEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0