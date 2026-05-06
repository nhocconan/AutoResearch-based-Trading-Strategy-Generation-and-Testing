#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 4h Donchian(20) breakout + volume confirmation
# Choppiness Index identifies ranging (chop > 61.8) vs trending (chop < 38.2) markets
# In trending regime (chop < 38.2), trade Donchian breakouts with volume confirmation
# In ranging regime (chop > 61.8), fade reversals at Donchian channels with volume confirmation
# Uses only 4h timeframe + volume + chop filter to minimize overtrading
# Target: 20-50 trades/year to avoid fee drag; works in bull/bear via regime adaptation

name = "4h_ChopRegime_Donchian20_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss and chop calculation
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(high, period) - min(low, period))) / log10(period)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Calculate Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(chop[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(atr[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Regime identification
        is_trending = chop[i] < 38.2   # Trending regime
        is_ranging = chop[i] > 61.8    # Ranging regime
        
        if position == 0:
            if is_trending:
                # Trending regime: trade breakouts
                # Long breakout: price > upper channel AND volume spike
                if close[i] > highest_high_20[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                    long_extreme = close[i]
                # Short breakdown: price < lower channel AND volume spike
                elif close[i] < lowest_low_20[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
                    short_extreme = close[i]
            elif is_ranging:
                # Ranging regime: fade at channels (mean reversion)
                # Long at support: price < lower channel AND volume spike
                if close[i] < lowest_low_20[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                    long_extreme = close[i]
                # Short at resistance: price > upper channel AND volume spike
                elif close[i] > highest_high_20[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
                    short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit conditions
            exit_signal = False
            if is_trending:
                # In trending regime, exit on 30% ATR retracement from extreme
                if close[i] <= long_extreme - 0.3 * atr[i]:
                    exit_signal = True
            else:
                # In ranging regime, exit at opposite channel or midpoint
                midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
                if close[i] >= midpoint:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit conditions
            exit_signal = False
            if is_trending:
                # In trending regime, exit on 30% ATR retracement from extreme
                if close[i] >= short_extreme + 0.3 * atr[i]:
                    exit_signal = True
            else:
                # In ranging regime, exit at opposite channel or midpoint
                midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
                if close[i] <= midpoint:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals