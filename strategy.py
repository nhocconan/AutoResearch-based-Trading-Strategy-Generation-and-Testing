#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + 1d Donchian breakout + volume confirmation
# - Primary signal: 6h ADX(14) > 25 indicates trending market
# - Direction filter: 1d Donchian breakout (price > 20-period high for long, < 20-period low for short)
# - Volume filter: 6h volume > 1.5x 20-period average volume (momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(14) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: ADX filters ranging markets; Donchian breakout captures trends in both directions

name = "6h_1d_adx_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Pre-compute 6h ADX(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # True Range
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_6h - np.roll(high_6h, 1)) > (np.roll(low_6h, 1) - low_6h), 
                       np.maximum(high_6h - np.roll(high_6h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_6h, 1) - low_6h) > (high_6h - np.roll(high_6h, 1)), 
                        np.maximum(np.roll(low_6h, 1) - low_6h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[di_plus + di_minus == 0] = 0
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    # Pre-compute 6h ATR(14) for stoploss
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX < 20 (trend weakening) OR stoploss hit
            if adx[i] < 20.0 or close_6h[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX < 20 (trend weakening) OR stoploss hit
            if adx[i] < 20.0 or close_6h[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with ADX and volume filters
            if vol_spike[i] and adx[i] > 25.0:
                # Long: price breaks above 1d Donchian high
                if close_6h[i] > donchian_high_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price breaks below 1d Donchian low
                elif close_6h[i] < donchian_low_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals