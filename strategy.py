#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h volume confirmation + 1d ADX trend filter
# - Primary signal: Donchian breakout on 6h - long when price > 20-bar high, short when price < 20-bar low
# - Volume filter: 12h volume > 20-period median volume (ensures participation)
# - Trend filter: 1d ADX > 25 (confirms trending market, avoids chop)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: ADX filter ensures we only trade in trending regimes, Donchian captures breakouts

name = "6h_12h_1d_donchian_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h volume regime: volume > 20-period median volume
    volume_12h = df_12h['volume'].values
    median_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    volume_regime_12h = volume_12h > median_volume_20
    volume_regime_aligned = align_htf_to_ltf(prices, df_12h, volume_regime_12h)
    
    # Pre-compute 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx[np.isnan(dx)] = 0  # Handle division by zero
    
    adx_regime = adx > 25  # Trending market
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_regime)
    
    # 6h price data for Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20) channels on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR ADX falls below 20 (trend weakening)
            if close[i] < donchian_low[i] or not adx_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR ADX falls below 20 (trend weakening)
            if close[i] > donchian_high[i] or not adx_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and ADX trend filter
            # Long: price breaks above Donchian high AND volume regime AND ADX > 25
            if close[i] > donchian_high[i] and volume_regime_aligned[i] and adx_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume regime AND ADX > 25
            elif close[i] < donchian_low[i] and volume_regime_aligned[i] and adx_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals