#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike confirmation and ATR trailing stop
# Uses 4h primary timeframe with 1d HTF for volume regime detection (spike = institutional interest).
# Donchian(20) breakout captures medium-term momentum with clear structure.
# Volume spike: 1d volume > 2.0x 20-period average confirms strong participation.
# ATR trailing stop (2.5x) protects gains and limits drawdown in choppy markets.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while maintaining edge.
# Works in bull markets via upside breakouts and in bear markets via downside breakouts during volume expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for volume regime) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 4h Donchian channels (20-period) ===
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # === 1d Volume spike filter ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * vol_ma_20_1d)  # True when volume spikes
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        spike_ok = vol_spike_aligned[i]
        
        # === ATR CALCULATION FOR TRAILING STOP ===
        atr_4h = np.abs(high_4h - low_4h)
        atr_ma = pd.Series(atr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
        atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from high
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from low
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Donchian breakout in opposite direction) ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low
            if price < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high
            if price > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike (institutional participation)
            if spike_ok:
                # Go long when price breaks above Donchian high (bullish breakout)
                if price > donch_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Go short when price breaks below Donchian low (bearish breakout)
                elif price < donch_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0