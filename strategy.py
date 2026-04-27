#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + Donchian Breakout with Volume Confirmation
# Choppiness Index (CHOP) identifies ranging vs trending markets
# CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending (trend following)
# Donchian(20) breakout in trending regime with volume confirmation
# In ranging regime, fade moves near Donchian bands (mean reversion)
# Uses volume > 1.3x 20-period average for confirmation
# Target: 20-40 trades/year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Choppiness Index (14-period)
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr_period = 14
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high - lowest_low
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(atr * 14 / range_14) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # Neutral when undefined
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop[i] < 38.2:  # Trending regime
            # Long: price breaks above Donchian high with volume confirmation and above daily EMA50
            if (close[i] > donchian_high[i] and volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low with volume confirmation and below daily EMA50
            elif (close[i] < donchian_low[i] and volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                # Hold position or exit if conditions weaken
                if position == 1:
                    # Exit long if price returns to middle of Donchian channel
                    mid = (donchian_high[i] + donchian_low[i]) / 2
                    if close[i] < mid:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.30
                elif position == -1:
                    # Exit short if price returns to middle of Donchian channel
                    mid = (donchian_high[i] + donchian_low[i]) / 2
                    if close[i] > mid:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.30
                else:
                    signals[i] = 0.0
                    
        elif chop[i] > 61.8:  # Ranging regime
            # Mean reversion near Donchian bands
            if position == 0:  # Only enter new positions when flat
                # Long near lower band
                if (close[i] <= donchian_low[i] * 1.002 and volume_filter[i] and 
                    close[i] > ema50_1d_aligned[i]):  # Slight buffer above band
                    signals[i] = 0.25
                    position = 1
                # Short near upper band
                elif (close[i] >= donchian_high[i] * 0.998 and volume_filter[i] and 
                      close[i] < ema50_1d_aligned[i]):  # Slight buffer below band
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Exit positions when price reaches opposite band or middle
                if position == 1:
                    if close[i] >= donchian_high[i] * 0.998:  # Near upper band
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                elif position == -1:
                    if close[i] <= donchian_low[i] * 1.002:  # Near lower band
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:  # Transition regime (38.2 <= CHOP <= 61.8)
            # Hold current position or stay flat
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Chop_Donchian_Volume_Regime"
timeframe = "4h"
leverage = 1.0