#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
    # Choppiness Index identifies ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets
    # In trending regimes: buy breakouts above Donchian high, sell breakdowns below Donchian low
    # In ranging regimes: fade moves at Donchian channels (sell near high, buy near low)
    # Volume confirmation ensures breakouts have conviction
    # This adapts to both bull/bear markets by following trend when present, mean-reverting in ranges
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]  # First TR is just high-low
    for i in range(1, len(close)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    atr14_safe = np.where(atr14 == 0, 1e-10, atr14)
    chop = 100 * np.log10((highest_high14 - lowest_low14) / atr14_safe / 14) / np.log10(14)
    
    # Donchian channels (20-period)
    donch_high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.3 * vol_ma20  # Volume surge filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after indicators are ready
        # Skip if data not ready
        if (np.isnan(chop[i]) or np.isnan(donch_high20[i]) or np.isnan(donch_low20[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        price = close[i]
        
        if position == 0:
            # Enter based on regime
            if chop_val < 38.2:  # Trending regime
                # Breakout long
                if price > donch_high20[i] and vol_surge[i]:
                    signals[i] = 0.25
                    position = 1
                # Breakdown short
                elif price < donch_low20[i] and vol_surge[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop_val > 61.8:  # Ranging regime
                # Fade at channels: sell near high, buy near low
                if price > 0.95 * donch_high20[i]:  # Near upper channel
                    signals[i] = -0.20
                    position = -1
                elif price < 1.05 * donch_low20[i]:  # Near lower channel
                    signals[i] = 0.20
                    position = 1
        else:
            # Exit conditions
            if position == 1:  # Long position
                # Exit trending: breakdown or chop becomes ranging
                if price < donch_low20[i] or chop_val > 50:
                    signals[i] = 0.0
                    position = 0
                # Exit ranging: price moves to middle
                elif chop_val > 61.8 and price > 0.5 * (donch_high20[i] + donch_low20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit trending: breakout or chop becomes ranging
                if price > donch_high20[i] or chop_val > 50:
                    signals[i] = 0.0
                    position = 0
                # Exit ranging: price moves to middle
                elif chop_val > 61.8 and price < 0.5 * (donch_high20[i] + donch_low20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "4h_Choppiness_Donchian_BreakoutFade_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0