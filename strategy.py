#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 4h Donchian(20) breakout + volume confirmation
# Uses Choppiness Index to identify trending vs ranging markets: only trade breakouts in trending regimes
# Choppiness > 61.8 = ranging (avoid), Choppiness < 38.2 = trending (trade breakouts)
# Combines with Donchian breakouts for trend continuation and volume filter to avoid false signals
# Designed for 4h timeframe with target of 75-200 trades over 4 years (19-50/year)
# Works in bull/bear markets by requiring trend regime alignment
name = "4h_Chop_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (14-period) - higher = more ranging, lower = more trending
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula
    chop = np.where(
        (atr > 0) & (hh - ll > 0),
        100 * np.log10(tr_sum / (atr * 14)) / np.log10(14),
        50  # default when undefined
    )
    
    # Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in trending markets (Choppiness < 38.2)
        trending_regime = chop[i] < 38.2
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above 20-period high
        short_breakout = close[i] < donchian_low[i-1]  # Break below 20-period low
        
        if position == 0:
            # Long: bullish breakout + trending regime + volume confirmation
            if long_breakout and trending_regime and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + trending regime + volume confirmation
            elif short_breakout and trending_regime and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout below Donchian low or regime change to ranging
            if close[i] < donchian_low[i] or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout above Donchian high or regime change to ranging
            if close[i] > donchian_high[i] or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals