#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation.
# Uses Choppiness Index (14) to detect ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In trending regimes: breakout entries in direction of trend.
# In ranging regimes: mean-reversion at Donchian channels.
# Volume confirmation required for all entries.
# Designed for 4h timeframe to balance trade frequency and signal quality.
# Target: 20-40 trades/year per symbol (80-160 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) for Choppiness Index
    tr1 = np.abs(np.roll(high_1d, 1) - np.roll(low_1d, 1))
    tr2 = np.abs(np.roll(high_1d, 1) - np.roll(close_1d, 1))
    tr3 = np.abs(np.roll(low_1d, 1) - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14)
    # Sum of true ranges over 14 periods
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range14 = hh14 - ll14
    chop = np.where(range14 > 0, 100 * np.log10(sum_tr14 / range14) / np.log10(14), 50)
    
    # Donchian channel (20) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Align indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Trending market (CHOP < 38.2): breakout in direction of trend
            if chop_val < 38.2:
                # Determine trend direction using price vs 50-period EMA
                ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
                ema50_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema50)
                trend_up = close[i] > ema50_aligned[i]
                
                if trend_up and close[i] > high_20_aligned[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif not trend_up and close[i] < low_20_aligned[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (CHOP > 61.8): mean reversion at Donchian channels
            elif chop_val > 61.8:
                if close[i] <= low_20_aligned[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= high_20_aligned[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price reaches upper Donchian or chop signals strong trend
                if close[i] >= high_20_aligned[i] or chop_val < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price reaches lower Donchian or chop signals strong trend
                if close[i] <= low_20_aligned[i] or chop_val < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Regime_Donchian20_Breakout_Volume"
timeframe = "4h"
leverage = 1.0