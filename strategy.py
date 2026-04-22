#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation
    # Choppiness Index > 61.8 = ranging market (mean reversion at Donchian bands)
    # Choppiness Index < 38.2 = trending market (breakout continuation)
    # In ranging markets: fade Donchian breakouts (sell at upper band, buy at lower band)
    # In trending markets: follow Donchian breakouts (buy breakouts, sell breakdowns)
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: adapts to market regime
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14-period)
    atr = pd.Series(np.maximum(high - low,
                               np.maximum(np.abs(high - np.roll(close, 1)),
                                          np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr[0] = high[0] - low[0]
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(atr * 14 / range_hl) / np.log10(14)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine market regime based on Choppiness Index
            if chop[i] > 61.8:  # Ranging market - mean reversion
                # Sell at upper Donchian band, buy at lower Donchian band
                if close[i] >= high_max[i] and vol_spike[i]:
                    signals[i] = -0.25  # Short at resistance
                    position = -1
                elif close[i] <= low_min[i] and vol_spike[i]:
                    signals[i] = 0.25   # Long at support
                    position = 1
            else:  # Trending market (chop < 38.2) or transition - follow breakouts
                # Buy breakouts, sell breakdowns
                if close[i] > high_max[i] and vol_spike[i]:
                    signals[i] = 0.25   # Long breakout
                    position = 1
                elif close[i] < low_min[i] and vol_spike[i]:
                    signals[i] = -0.25  # Short breakdown
                    position = -1
        else:
            # Exit conditions
            if position == 1:  # Long position
                # Exit on reversal to opposite Donchian band or loss of momentum
                if close[i] < low_min[i]:  # Reached opposite band
                    signals[i] = 0.0
                    position = 0
                elif chop[i] > 61.8 and close[i] < high_max[i] * 0.995:  # Lost momentum in ranging market
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit on reversal to opposite Donchian band or loss of momentum
                if close[i] > high_max[i]:  # Reached opposite band
                    signals[i] = 0.0
                    position = 0
                elif chop[i] > 61.8 and close[i] > low_min[i] * 1.005:  # Lost momentum in ranging market
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Chop_Regime_Donchian20_BreakoutFade_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0