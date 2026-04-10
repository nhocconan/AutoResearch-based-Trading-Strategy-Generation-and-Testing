#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR-based trailing stop
# - Long when price breaks above 20-period Donchian high + 1d volume > 1.5x 20-day average
# - Short when price breaks below 20-period Donchian low + 1d volume > 1.5x 20-day average
# - Uses 1d HTF for volume confirmation to ensure institutional participation
# - ATR-based trailing stop: exit when price moves 2.5x ATR against position from extreme
# - Designed for 4h timeframe: targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: breakouts capture momentum in both regimes
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "4h_1d_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 4h Donchian channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    extreme_price = 0.0  # Tracks highest high for long, lowest low for short
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update extreme price (highest high since entry)
            if high_4h[i] > extreme_price:
                extreme_price = high_4h[i]
            
            # Exit: ATR-based trailing stop or price re-enters Donchian channel
            if (close_4h[i] < extreme_price - 2.5 * atr_14[i] or 
                close_4h[i] < donchian_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update extreme price (lowest low since entry)
            if low_4h[i] < extreme_price:
                extreme_price = low_4h[i]
            
            # Exit: ATR-based trailing stop or price re-enters Donchian channel
            if (close_4h[i] > extreme_price + 2.5 * atr_14[i] or 
                close_4h[i] > donchian_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above Donchian high
                if close_4h[i] > donchian_high[i]:
                    position = 1
                    entry_price = close_4h[i]
                    extreme_price = high_4h[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low
                elif close_4h[i] < donchian_low[i]:
                    position = -1
                    entry_price = close_4h[i]
                    extreme_price = low_4h[i]
                    signals[i] = -0.25
    
    return signals