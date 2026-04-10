#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1d ADX regime filter
# - Primary: 12h price breaking above/below 20-period Donchian channels captures medium-term momentum
# - Volume filter: 1d volume > 1.8x 20-period volume MA confirms institutional participation
# - Regime filter: 1d ADX(14) > 20 ensures trending market (avoids choppy ranging conditions)
# - Exit: Price reverses back to opposite Donchian channel (middle or opposite band)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volume confirms strength, ADX filters weak trends
# - Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe

name = "12h_1d_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    # Middle band = (upper + lower) / 2
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d volume spike filter: volume > 1.8x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d ADX(14) for regime filter
    high_diff_1d = high_1d - np.roll(high_1d, 1)
    low_diff_1d = np.roll(low_1d, 1) - low_1d
    close_diff_1d = np.roll(close_1d, 1) - close_1d
    high_diff_1d[0] = 0
    low_diff_1d[0] = 0
    close_diff_1d[0] = 0
    
    plus_dm_1d = np.where((high_diff_1d > low_diff_1d) & (high_diff_1d > 0), high_diff_1d, 0)
    minus_dm_1d = np.where((low_diff_1d > high_diff_1d) & (low_diff_1d > 0), low_diff_1d, 0)
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = high_1d[0] - low_1d[0]
    tr2_1d[0] = np.abs(high_1d[0] - close_1d[0])
    tr3_1d[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_14_1d = pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).mean().values
    minus_dm_14_1d = pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).mean().values
    
    plus_di_1d = np.where(atr_14_1d > 0, 100 * plus_dm_14_1d / atr_14_1d, 0)
    minus_di_1d = np.where(atr_14_1d > 0, 100 * minus_dm_14_1d / atr_14_1d, 0)
    
    dx_1d = np.where((plus_di_1d + minus_di_1d) > 0, 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.8x 20-period volume MA
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_spike = volume_1d_current[i] > 1.8 * volume_ma_20_1d_aligned[i]
        
        # Regime filter: ADX > 20 to ensure trending conditions
        strong_trend = adx_1d_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band + vol spike + strong trend
            if (close[i] > donchian_upper[i] and 
                vol_spike and strong_trend):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band + vol spike + strong trend
            elif (close[i] < donchian_lower[i] and 
                  vol_spike and strong_trend):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price reverses back to Donchian middle band (opposite direction)
            if position == 1:  # Long position
                if close[i] < donchian_middle[i]:  # Exit when price crosses below middle band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > donchian_middle[i]:  # Exit when price crosses above middle band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals