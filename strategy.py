#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 4h Donchian breakout and volume confirmation
# Uses Choppiness Index (14) to identify trending (CHOP < 38.2) vs ranging (CHOP > 61.8) markets.
# In trending regimes: trade Donchian(20) breakouts with volume confirmation.
# In ranging regimes: fade moves at Donchian bands with RSI(14) extremes.
# Designed to reduce whipsaws and adapt to changing market conditions.
name = "4h_Chop_Donchian20_Volume_Adaptive"
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
    
    # Calculate ATR for Donchian and Choppiness
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * ATR)) / log10(n)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (np.log10(sum_tr14) - np.log10(14 * atr)) / np.log10(14)
    
    # Volume filter: current volume > 1.3x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * avg_volume)
    
    # RSI for mean reversion in ranging markets
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_filter[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        if position == 0:
            # Enter based on regime
            if is_trending and volume_filter[i]:
                # Trending: trade breakouts
                if close[i] > highest_high[i-1]:  # Break above Donchian high
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low[i-1]:  # Break below Donchian low
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging: mean reversion at extremes
                if close[i] <= lowest_low[i] and rsi[i] < 30:  # At support + oversold
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= highest_high[i] and rsi[i] > 70:  # At resistance + overbought
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long
            if is_trending:
                # In trending: exit on Donchian low break or trend weakening
                if close[i] < lowest_low[i] or chop[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging: exit on mean reversion or range breakdown
                if close[i] >= highest_high[i] or rsi[i] > 70 or chop[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short
            if is_trending:
                # In trending: exit on Donchian high break or trend weakening
                if close[i] > highest_high[i] or chop[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging: exit on mean reversion or range breakdown
                if close[i] <= lowest_low[i] or rsi[i] < 30 or chop[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals