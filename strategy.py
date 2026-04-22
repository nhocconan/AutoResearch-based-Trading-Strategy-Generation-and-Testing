#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + Daily Donchian(20) breakout with volume confirmation
# Choppiness Index (CHOP) identifies ranging vs trending markets: CHOP > 61.8 = range (mean-revert),
# CHOP < 38.2 = trend (follow breakouts). In ranging markets, we fade Donchian breakouts (sell breaks above upper band, buy breaks below lower band).
# In trending markets, we follow breakouts (buy breaks above upper band, sell breaks below lower band).
# Uses daily timeframe for Donchian channels and CHOP calculation for stability, with 12h execution.
# Volume spike (>1.5x 20-period average) confirms breakout strength. Designed for low trade frequency (~15-30/year) to minimize fee decay.
# Works in both bull and bear markets by adapting to market regime.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data for Donchian channels and Choppiness Index (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper band = 20-period high, Lower band = 20-period low
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index on daily data (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # First period
    tr3[0] = tr1[0]  # First period
    atr_14 = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop_raw, 50.0)  # Default to 50 (neutral) when range is zero
    
    # Align daily indicators to 12h timeframe (waits for daily bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period average volume for volume spike detection (12h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        chop_val = chop_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Regime-based entry logic
            if chop_val < 38.2:  # Trending market - follow breakouts
                # Long: price breaks above upper Donchian band
                if price > upper and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian band
                elif price < lower and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif chop_val > 61.8:  # Ranging market - fade breakouts (mean reversion)
                # Short: price breaks above upper Donchian band (sell the breakout)
                if price > upper and vol_spike:
                    signals[i] = -0.25
                    position = -1
                # Long: price breaks below lower Donchian band (buy the breakout)
                elif price < lower and vol_spike:
                    signals[i] = 0.25
                    position = 1
        
        elif position != 0:
            # Exit conditions: opposite band touch or volatility expansion
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price touches lower Donchian band (mean reversion) or volatility expands
                if price < lower:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price touches upper Donchian band (mean reversion) or volatility expands
                if price > upper:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Chop_Donchian20_Breakout_Volume"
timeframe = "12h"
leverage = 1.0