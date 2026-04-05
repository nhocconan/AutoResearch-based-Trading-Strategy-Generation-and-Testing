#!/usr/bin/env python3
"""
Experiment #9521: 4h Donchian Breakout + Volume + Regime Filter (1d/1w)
Hypothesis: Donchian channel (20-period) breakouts on 4h timeframe, confirmed by volume spikes 
and regime filtering (Choppiness Index > 61.8 for mean reversion at channel edges, 
Choppiness Index < 38.2 for trend continuation breakouts), provide robust performance 
in both bull and bear markets. Targets 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9521_4h_donchian_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
CHOP_PERIOD = 14
CHOP_THRESHOLD_HIGH = 61.8  # >61.8 = ranging/choppy
CHOP_THRESHOLD_LOW = 38.2   # <38.2 = trending
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (EWM with alpha=1/period)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    hh = pd.Series(high).rolling(window=period, min_periods=period).max()
    ll = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = np.where((hh - ll) != 0,
                    100 * np.log10(atr_sum / (hh - ll)) / np.log10(period),
                    50)
    return chop.values

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max()
    lower = pd.Series(low).rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d and 1w for regime context)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d close for trend filter
    close_1d = df_1d['close'].values
    close_1d_ma = pd.Series(close_1d).rolling(window=10, min_periods=10).mean().values
    close_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, close_1d_ma)
    
    # Calculate 1w trend filter (optional - can be used for regime)
    close_1w = df_1w['close'].values
    close_1w_ma = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values
    close_1w_ma_aligned = align_htf_to_ltf(prices, df_1w, close_1w_ma)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_up, donch_low = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Choppiness Index for regime filtering
    chop = calculate_choppiness(high, low, close, CHOP_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if (np.isnan(donch_up[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER)
        
        # Regime filters based on Choppiness
        ranging_market = chop[i] > CHOP_THRESHOLD_HIGH   # Chop > 61.8 = ranging
        trending_market = chop[i] < CHOP_THRESHOLD_LOW   # Chop < 38.2 = trending
        
        # Mean reversion in ranging markets: fade at channel edges
        mean_rev_long = ranging_market and volume_spike and close[i] <= donch_low[i]
        mean_rev_short = ranging_market and volume_spike and close[i] >= donch_up[i]
        
        # Trend continuation in trending markets: breakout in direction of trend
        # Use 1d MA to determine trend direction
        trend_up = not np.isnan(close_1d_ma_aligned[i]) and close[i] > close_1d_ma_aligned[i]
        trend_down = not np.isnan(close_1d_ma_aligned[i]) and close[i] < close_1d_ma_aligned[i]
        
        breakout_long = trending_market and volume_spike and close[i] >= donch_up[i] and trend_up
        breakout_short = trending_market and volume_spike and close[i] <= donch_low[i] and trend_down
        
        # Entry conditions
        long_entry = mean_rev_long or breakout_long
        short_entry = mean_rev_short or breakout_short
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals