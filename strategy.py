#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Choppiness Index regime filter + 4h Donchian(20) breakout + volume spike
    # Choppiness Index (CHOP) identifies market regime: >61.8 = ranging (mean revert), <38.2 = trending
    # In trending regimes (CHOP < 38.2), we trade Donchian breakouts with volume confirmation
    # In ranging regimes (CHOP > 61.8), we fade the extremes (sell at upper band, buy at lower band)
    # This adaptive approach works in both bull and bear markets by switching strategy based on regime
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (14-period)
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    range_max_min = hh - ll
    range_safe = np.where(range_max_min == 0, 1e-10, range_max_min)
    
    # Choppiness Index: 100 * log10(sum(ATR)/range) / log10(period)
    chop = 100 * np.log10(atr_safe * 14 / range_safe) / np.log10(14)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trending market (CHOP < 38.2): trade breakouts
            if chop[i] < 38.2:
                # Long: break above Donchian high with volume spike
                if close[i] > donchian_high[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below Donchian low with volume spike
                elif close[i] < donchian_low[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (CHOP > 61.8): fade extremes
            elif chop[i] > 61.8:
                # Short at upper Donchian band (overbought in range)
                if close[i] > donchian_high[i]:
                    signals[i] = -0.25
                    position = -1
                # Long at lower Donchian band (oversold in range)
                elif close[i] < donchian_low[i]:
                    signals[i] = 0.25
                    position = 1
        else:
            # Exit conditions
            if position == 1:  # Long position
                # Exit trending long: return to Donchian low
                if chop[i] < 38.2 and close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit ranging long: return to midpoint or stop loss
                elif chop[i] > 61.8 and close[i] > (donchian_high[i] + donchian_low[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit trending short: return to Donchian high
                if chop[i] < 38.2 and close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit ranging short: return to midpoint
                elif chop[i] > 61.8 and close[i] < (donchian_high[i] + donchian_low[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Chop_Regime_Donchian20_BreakoutFade_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0