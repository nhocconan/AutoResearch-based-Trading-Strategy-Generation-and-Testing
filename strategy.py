#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRTrail
Hypothesis: Donchian(20) breakout on 6h timeframe with daily EMA50 trend filter and volume spike confirmation (>1.5x average volume).
Only trade breakouts aligned with daily EMA50 direction during volume expansion.
Exit via ATR-based trailing stop (3.0*ATR from extreme) to capture trends while limiting drawdown.
Designed to work in both bull (trend-following breakouts) and bear (short breakdowns) by following the daily trend.
Uses discrete sizing (0.25) and targets 12-30 trades/year on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate ATR(14) for 6h (used for stoploss and Donchian)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14, 20)  # EMA needs 50, vol needs 20, ATR needs 14, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or 
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
        ema_val = ema_1d_aligned[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = vol_val > 1.5 * vol_ma_val
        
        # Donchian(20) breakout levels
        lookback_start = max(0, i - 19)
        highest_20 = np.max(high[lookback_start:i+1])
        lowest_20 = np.min(low[lookback_start:i+1])
        
        if position == 0:
            # Look for entry signals: Donchian breakout with trend and volume confirmation
            # Long: price breaks above 20-period high, above daily EMA50, with volume spike
            long_signal = (high_val > highest_20) and (close_val > ema_val) and volume_spike
            # Short: price breaks below 20-period low, below daily EMA50, with volume spike
            short_signal = (low_val < lowest_20) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_high_since_entry = high_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_low_since_entry = low_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high_val)
            # ATR-based trailing stop: exit if price drops 3.0*ATR from highest high
            if close_val < highest_high_since_entry - 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high_since_entry = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low_val)
            # ATR-based trailing stop: exit if price rises 3.0*ATR from lowest low
            if close_val > lowest_low_since_entry + 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_low_since_entry = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRTrail"
timeframe = "6h"
leverage = 1.0