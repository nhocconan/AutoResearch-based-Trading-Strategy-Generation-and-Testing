#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volume spike filter and chop regime
# - Long when price breaks above 4h Donchian upper (20-bar high) AND 1d volume > 2.0x 20-day average
# - Short when price breaks below 4h Donchian lower (20-bar low) AND 1d volume > 2.0x 20-day average
# - Chop regime filter: 4h Choppiness Index > 61.8 = ranging (fade Donchian touches)
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Novelty: Combines Donchian breakout with 1d volume confirmation (more reliable than lower TF volume)
#            and chop regime to avoid false breakouts in sideways markets
# - Works in both bull/bear: Donchian adapts to volatility, volume confirms institutional interest,
#                            chop filter prevents whipsaws in ranging markets

name = "4h_1d_donchian_volume_chop_v1"
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
    
    # Pre-compute 1d indicators
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume > 2.0x 20-day average (volume confirmation)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20)
    
    # Align 1d volume spike to 4h timeframe (completed 1d bar only)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,n) - min(low,n))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14 / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    chop_regime = chop > 61.8  # True when ranging/markets suitable for mean reversion
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_regime[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR Donchian lower touch (mean reversion in chop)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               (chop_regime[i] and low[i] <= donchian_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR Donchian upper touch (mean reversion in chop)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               (chop_regime[i] and high[i] >= donchian_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with 1d volume confirmation
            # Long: price breaks above Donchian upper AND 1d volume spike
            if high[i] >= donchian_high[i] and volume_spike_aligned[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below Donchian lower AND 1d volume spike
            elif low[i] <= donchian_low[i] and volume_spike_aligned[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals