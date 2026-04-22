#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Choppiness Index regime filter with Donchian(20) breakout and volume confirmation
    # Chop > 61.8 = range-bound (fade extremes), Chop < 38.2 = trending (follow breakouts)
    # Donchian breakouts provide clear entry/exit levels
    # Volume spike confirms institutional participation
    # Works in bull/bear by adapting to market regime
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (14-period) on 4h data
    atr = np.zeros(n)
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((hh - ll) / (atr_safe * 14)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)  # Handle case where range is zero
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # In trending market (Chop < 38.2): follow breakouts
            if chop[i] < 38.2:
                # Long: Break above Donchian high with volume spike
                if close[i] > donchian_high[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below Donchian low with volume spike
                elif close[i] < donchian_low[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # In ranging market (Chop > 61.8): fade extremes
            elif chop[i] > 61.8:
                # Long: Near Donchian low with volume spike (mean reversion)
                if close[i] < donchian_low[i] * 1.02 and vol_spike[i]:  # Within 2% of low
                    signals[i] = 0.25
                    position = 1
                # Short: Near Donchian high with volume spike (mean reversion)
                elif close[i] > donchian_high[i] * 0.98 and vol_spike[i]:  # Within 2% of high
                    signals[i] = -0.25
                    position = -1
            # In transition zone (38.2 <= Chop <= 61.8): no new positions
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below Donchian low OR chop becomes too high (trend weakening)
                if close[i] < donchian_low[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price crosses above Donchian high OR chop becomes too high (trend weakening)
                if close[i] > donchian_high[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Chop_Regime_Donchian20_Breakout_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0