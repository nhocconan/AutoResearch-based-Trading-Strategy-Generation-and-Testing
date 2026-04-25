#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRTrail
Hypothesis: On 12h timeframe, Donchian(20) breakouts with 1d EMA50 trend filter and volume spike (>2.0x 20-bar avg) captures strong institutional moves with low trade frequency. ATR-based trailing stop (3.0x ATR) manages risk. Designed for 12-37 trades/year to minimize fee drag. Works in both bull and bear markets via trend filter and volatility-based exit.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility and trailing stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14)  # EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with trend and volume
            # Long: price breaks above Donchian high with uptrend (close > EMA50) and volume spike
            long_signal = (high_val > donch_high) and (close_val > ema_val) and volume_spike
            # Short: price breaks below Donchian low with downtrend (close < EMA50) and volume spike
            short_signal = (low_val < donch_low) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 3.0x ATR from highest since entry
            if close_val < highest_since_entry - 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 3.0x ATR from lowest since entry
            if close_val > lowest_since_entry + 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0