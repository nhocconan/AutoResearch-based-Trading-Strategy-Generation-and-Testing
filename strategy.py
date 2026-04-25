#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopRegime_1dTrend
Hypothesis: Donchian(20) breakouts on 4h with volume spike confirmation and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend) aligned with 1d EMA50 trend. In trending regimes (CHOP < 38.2), follow breakout direction with 1d trend. In ranging regimes (CHOP > 61.8), mean revert at Donchian bands. Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index: 100 * log10(sum(ATR) / (n * (max(high) - min(low)))) / log10(n)"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    range_hl = max_high - min_low
    
    chop = np.where(
        (range_hl > 0) & (period > 0),
        100 * np.log10(atr_sum / (range_hl * period)) / np.log10(period),
        50.0  # neutral when range is zero
    )
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 4h Choppiness Index (14-period)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + volume MA (20) + EMA (50) + chop (14)
    start_idx = max(20, 20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals
            long_breakout = curr_high > donchian_high[i]
            short_breakout = curr_low < donchian_low[i]
            
            # Regime-based logic
            if chop[i] < 38.2:  # Trending regime
                # Follow breakout direction with 1d trend filter
                long_entry = long_breakout and volume_spike[i] and (curr_close > ema_50_1d_aligned[i])
                short_entry = short_breakout and volume_spike[i] and (curr_close < ema_50_1d_aligned[i])
            elif chop[i] > 61.8:  # Ranging regime
                # Mean revert at Donchian bands
                long_entry = short_breakout and volume_spike[i] and (curr_close < ema_50_1d_aligned[i])  # short breakout -> long mean reversion
                short_entry = long_breakout and volume_spike[i] and (curr_close > ema_50_1d_aligned[i])   # long breakout -> short mean reversion
            else:  # Neutral regime (38.2 <= CHOP <= 61.8)
                # No clear edge, stay flat
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
            # Long position: exit when price closes below Donchian low or trend turns bearish in trending regime
            if chop[i] < 38.2:  # Trending regime
                if curr_close < donchian_low[i] or curr_close < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging or neutral regime: exit at mean reversion
                if curr_close > (donchian_high[i] + donchian_low[i]) / 2:  # midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Donchian high or trend turns bullish in trending regime
            if chop[i] < 38.2:  # Trending regime
                if curr_close > donchian_high[i] or curr_close > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging or neutral regime: exit at mean reversion
                if curr_close < (donchian_high[i] + donchian_low[i]) / 2:  # midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime_1dTrend"
timeframe = "4h"
leverage = 1.0