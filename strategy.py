#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with volume confirmation and chop regime filter
# 1w Donchian channels provide strong structural levels that work in both bull and bear markets
# Volume confirmation (current 1d volume > 1.5x 20-period average) filters false breakouts
# Chop regime filter: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion at extremes
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_h_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_l_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_h_aligned = align_htf_to_ltf(prices, df_1w, donchian_h_20)
    donchian_l_aligned = align_htf_to_ltf(prices, df_1w, donchian_l_20)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute chop regime filter on 1d timeframe
    # Chop = 100 * log10(sum(ATR(1)) / (log10(n) * (highest_high - lowest_low)))
    # Simplified: chop > 61.8 = ranging, chop < 38.2 = trending
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # first bar
    atr1 = pd.Series(tr).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high - lowest_low
    chop = np.where(range_14 > 0, 100 * np.log10(sum_atr1 / (np.log10(14) * range_14)), 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_h_aligned[i]) or np.isnan(donchian_l_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: only trade in ranging markets (chop > 61.8)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit on Donchian lower band retracement (mean reversion)
            if close[i] < donchian_l_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on Donchian upper band retracement (mean reversion)
            if close[i] > donchian_h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion at extremes with volume confirmation and ranging market
            # Short on Donchian upper band touch, Long on Donchian lower band touch
            if volume_confirmed and ranging_market:
                if close[i] > donchian_h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                elif close[i] < donchian_l_aligned[i]:
                    position = 1
                    signals[i] = 0.25
    
    return signals