#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRTrail_VolumeSpike_v1
Hypothesis: Donchian(20) breakouts on 4h with volume spike confirmation (>2.0x 20-bar avg volume) and ATR-based trailing stop. Uses 1d EMA50 as trend filter to align with higher timeframe momentum. Discrete sizing (0.25) limits trades to ~20-30/year to minimize fee drag. Designed for BTC/ETH robustness: trend filter reduces whipsaws in ranging markets, volume confirmation ensures breakout validity, and ATR trailing stop adapts to volatility. Works in bull markets via breakout continuation and in bear markets via trend-filtered short signals.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) for trailing stop and volatility normalization
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-bar average volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need warmup for Donchian, ATR, volume MA, and EMA50
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma20[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-bar average
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Long: price breaks above Donchian high with volume spike and above 1d EMA50 (uptrend)
            # Short: price breaks below Donchian low with volume spike and below 1d EMA50 (downtrend)
            long_signal = (close[i] > donch_high[i]) and volume_confirm and (close[i] > ema50_1d_aligned[i])
            short_signal = (close[i] < donch_low[i]) and volume_confirm and (close[i] < ema50_1d_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position and update highest high
            signals[i] = 0.25
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # ATR trailing stop: exit when price drops 2.5*ATR from highest high since entry
            if high[i] >= lowest_low_since_entry:  # prevent invalid calculation when flipped
                exit_level = highest_high_since_entry - 2.5 * atr[i]
                if close[i] < exit_level:
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Short: hold position and update lowest low
            signals[i] = -0.25
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # ATR trailing stop: exit when price rises 2.5*ATR from lowest low since entry
            if low[i] <= highest_high_since_entry:  # prevent invalid calculation when flipped
                exit_level = lowest_low_since_entry + 2.5 * atr[i]
                if close[i] > exit_level:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATRTrail_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0