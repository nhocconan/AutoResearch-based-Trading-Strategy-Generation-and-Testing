#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index Regime + 12h Donchian Breakout + Volume Confirmation
# Choppiness Index (CHOP) identifies market regime: CHOP > 61.8 = ranging (mean revert),
# CHOP < 38.2 = trending (follow breakouts). In ranging markets, fade Donchian breakouts;
# in trending markets, continue breakouts. Uses 12h Donchian for structure and 6h for timing.
# Volume spike (1.5x 24-period avg) confirms momentum. Discrete sizing 0.25 minimizes fees.
# Targets 12-35 trades/year (50-140 total over 4 years) for 6h timeframe.
# Works in bull markets via trend continuation and in bear markets via mean reversion in ranges.

name = "6h_Chop_Regime_12hDonchian_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Donchian breakout
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high_12h = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_12h = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_12h = align_htf_to_ltf(prices, df_12h, highest_high_12h)
    donchian_low_12h = align_htf_to_ltf(prices, df_12h, lowest_low_12h)
    
    # Calculate 6h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(N)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First bar TR
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high - min_low
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # Avoid division by zero
    chop = 100 * (np.log10(sum_atr1) - np.log10(chop_denom)) / np.log10(14)
    
    # Calculate volume spike (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for CHOP and Donchian)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Determine regime: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
            if chop[i] > 61.8:  # Ranging market - mean revert Donchian breakouts
                # Long: price breaks below Donchian low + volume spike (fade breakdown)
                if close[i] < donchian_low_12h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks above Donchian high + volume spike (fade breakout)
                elif close[i] > donchian_high_12h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif chop[i] < 38.2:  # Trending market - continue Donchian breakouts
                # Long: price breaks above Donchian high + volume spike
                if close[i] > donchian_high_12h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low + volume spike
                elif close[i] < donchian_low_12h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # Choppy transition zone (38.2 <= CHOP <= 61.8) - no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: opposite Donchian breakout or regime shift to ranging
            if close[i] < donchian_low_12h[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: opposite Donchian breakout or regime shift to ranging
            if close[i] > donchian_high_12h[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals