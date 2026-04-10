#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume regime filter + ATR trailing stop
# - Long: price > Donchian high(20) AND 1d volume > 1.2x 20-day average (volume regime)
# - Short: price < Donchian low(20) AND 1d volume > 1.2x 20-day average
# - Uses ATR-based trailing stop: exit long when price drops 2.5x ATR from highest high since entry
# - Uses ATR-based trailing stop: exit short when price rises 2.5x ATR from lowest low since entry
# - Discrete position sizing (0.25) to minimize fee churn
# - Volume regime filter avoids low-volume false breakouts
# - Works in bull/bear markets: Donchian breakouts capture trends, volume filter ensures conviction
# - Target: 20-40 trades/year to avoid fee drag

name = "4h_1d_donchian_volume_regime_v1"
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
    
    # Pre-compute 1d volume regime (volume > 1.2x 20-day average)
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume_1d > (1.2 * avg_volume_20)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for trailing stop
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_regime_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high_4h[i])
            # Exit: price < Donchian low OR trailing stop hit
            if close_4h[i] < donchian_low[i] or close_4h[i] < highest_since_entry - 2.5 * atr_14[i]:
                position = 0
                highest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low_4h[i])
            # Exit: price > Donchian high OR trailing stop hit
            if close_4h[i] > donchian_high[i] or close_4h[i] > lowest_since_entry + 2.5 * atr_14[i]:
                position = 0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume regime filter
            if vol_regime_aligned[i]:
                # Long: price > Donchian high(20)
                if close_4h[i] > donchian_high[i]:
                    position = 1
                    entry_price = close_4h[i]
                    highest_since_entry = high_4h[i]
                    signals[i] = 0.25
                # Short: price < Donchian low(20)
                elif close_4h[i] < donchian_low[i]:
                    position = -1
                    entry_price = close_4h[i]
                    lowest_since_entry = low_4h[i]
                    signals[i] = -0.25
    
    return signals