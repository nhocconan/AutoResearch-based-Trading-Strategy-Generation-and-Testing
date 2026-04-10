#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w ADX regime filter
# - Primary: 12h Donchian breakout for clear structure-based entries
# - Volume filter: 1d volume > 1.3x 20-period volume MA to confirm participation
# - Regime filter: 1w ADX(14) > 20 to avoid choppy markets and ensure trending conditions
# - Exit: Price crosses opposite Donchian band (10-period for faster exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# - Works in bull/bear: Donchian captures breakouts, volume confirms strength, ADX avoids whipsaws

name = "12h_1d_1w_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 12h Donchian channels (20-period for entry, 10-period for exit)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 1d volume MA(20) for volume filter
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    # Calculate 1w ADX(14) for regime filter
    high_diff = high_1w - np.roll(high_1w, 1)
    low_diff = np.roll(low_1w, 1) - low_1w
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = np.where(atr_14 > 0, 100 * plus_dm_14 / atr_14, 0)
    minus_di = np.where(atr_14 > 0, 100 * minus_dm_14 / atr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(highest_10[i]) or np.isnan(lowest_10[i]) or
            np.isnan(volume_ma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.3x 20-period volume MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmed = volume_1d_aligned[i] > 1.3 * volume_ma_20_aligned[i]
        
        # Regime filter: ADX > 20 to avoid choppy markets
        trending = adx_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price above Donchian(20) high + volume confirmation + trending
            if (close[i] > highest_20[i] and volume_confirmed and trending):
                position = 1
                signals[i] = 0.25
            # Short entry: price below Donchian(20) low + volume confirmation + trending
            elif (close[i] < lowest_20[i] and volume_confirmed and trending):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price crosses opposite Donchian(10) band
            if position == 1:  # Long position
                if close[i] < lowest_10[i]:  # Exit when price crosses below Donchian(10) low
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > highest_10[i]:  # Exit when price crosses above Donchian(10) high
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals