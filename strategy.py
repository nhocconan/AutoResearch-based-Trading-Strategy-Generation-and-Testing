#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme Reversal with 1d Volume Spike and ADX Trend Filter
# Long when: Williams %R(14) crosses above -80 (oversold) AND 1d volume > 1.5x 20-period average AND 1d ADX(14) > 25 (trending market)
# Short when: Williams %R(14) crosses below -20 (overbought) AND 1d volume > 1.5x 20-period average AND 1d ADX(14) > 25 (trending market)
# Exit when: Williams %R returns to -50 (mean reversion) OR adverse move of 2x ATR(14)
# Williams %R captures extreme reversals, volume spike confirms conviction, ADX ensures trending environment
# Works in both bull and bear markets by trading mean reversals within established trends
# Target: 80-160 total trades over 4 years (20-40/year) with discrete sizing 0.25

name = "4h_WilliamsR_VolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume spike and ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX(14) and volume MA(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate 1d ADX components
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = minus_dm[0] = np.nan
    
    # Smoothed values for ADX
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    plus_di = np.where(atr_1d != 0, 100 * plus_dm_smooth / atr_1d, 0)
    minus_di = np.where(atr_1d != 0, 100 * minus_dm_smooth / atr_1d, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume spike filter (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Williams %R on 4h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when highest == lowest
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0, williams_r, -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX trend filter: trending market (ADX > 25)
        trending_market = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) in trending market with volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and trending_market and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) in trending market with volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and trending_market and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion)
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion)
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals