#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h 12h Donchian Breakout with Volume and Chop Filter v1
# Hypothesis: In ranging markets (2025), price often breaks Donchian channels with volume,
# then continues in the breakout direction. We use 12h Donchian for trend context and 4h for entry.
# Chop filter ensures we only trade in trending markets to avoid false breakouts in ranges.
# Works in both bull/bear by capturing breakouts with institutional volume.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_12h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend context (Donchian channel)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 20-period high and low for Donchian
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Chop filter: Choppy market indicator (EHLERS) - high = ranging, low = trending
    # We only trade when chop < 61.8 (trending market)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Calculate highest high and lowest low over ATR period
    hh = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max()
    ll = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min()
    
    # Chop formula: 100 * log10(sum(TR, atr_period) / (hh - ll)) / log10(atr_period)
    sum_tr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum()
    chop = 100 * np.log10(sum_tr / (hh - ll + 1e-10)) / np.log10(atr_period)
    chop_values = chop.values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(chop_values[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (chop < 61.8)
        if chop_values[i] >= 61.8:
            # In ranging markets, stay flat or close existing positions
            if position == 1 and close[i] <= donchian_low_4h[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] >= donchian_high_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position or stay flat
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches 12h Donchian low (stop) or shows weakness
            if close[i] <= donchian_low_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches 12h Donchian high (stop) or shows weakness
            if close[i] >= donchian_high_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long breakout: price closes above 12h Donchian high
                if close[i] > donchian_high_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below 12h Donchian low
                elif close[i] < donchian_low_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals