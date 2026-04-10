#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR-based volume spike and choppiness regime filter
# - Long when price breaks above 20-period Donchian high + 1d volume > 1.5x ATR(20)-scaled volume average + CHOP > 61.8 (range market)
# - Short when price breaks below 20-period Donchian low + same volume/CHOP conditions
# - Exit: price returns to midpoint of Donchian channel
# - Position sizing: 0.25 discrete level
# - Uses ATR-scaled volume threshold for adaptive confirmation (works in both high/low vol regimes)
# - 4h timeframe targets 20-50 trades/year with strict entry conditions

name = "4h_1d_donchian_atr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period Donchian channels on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d ATR(20) for adaptive volume threshold
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr1).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d volume average scaled by ATR (adaptive threshold)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_threshold_1d = vol_avg_1d * (atr_1d / np.mean(atr_1d[-100:]) if len(atr_1d) >= 100 else 1.0)
    vol_threshold_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold_1d)
    
    # Calculate Choppiness Index on 1d (14-period)
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hh = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1d['low'].rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop_raw = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    chop = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Current 1d volume (aligned)
    vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_threshold_1d_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_1d_current[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > ATR-scaled threshold (adaptive spike)
        vol_confirm = vol_1d_current[i] > vol_threshold_1d_aligned[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for breakout mean reversion)
        ranging_market = chop[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Exit conditions: return to midpoint of Donchian channel
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and ranging_market
        short_entry = short_breakout and vol_confirm and ranging_market
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals