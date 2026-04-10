#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Regime: ADX(14) from 1d > 25 = trending, < 20 = ranging
# - In trending regime (ADX>25): go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# - In ranging regime (ADX<20): fade extremes - long when Bear Power < 0 and turning up, short when Bull Power < 0 and turning down
# - Volume confirmation: current 6h volume > 1.5x 20-period average
# - Uses 6h timeframe to target 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Discrete position sizing (0.25) to minimize fee churn

name = "6h_1d_elder_ray_regime_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 1d Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = ema13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # Pre-compute 1d ADX(14) for regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h EMA13 for Elder Ray (for reference)
    close_6h = prices['close'].values
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive (momentum shift) or volume drops
            if bear_power_aligned[i] > 0 or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive (momentum shift) or volume drops
            if bull_power_aligned[i] > 0 or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries based on regime
            if vol_spike[i]:
                if adx_aligned[i] > 25:  # Trending regime
                    # Long: Bull Power positive and rising (bullish momentum)
                    # Short: Bear Power positive and rising (bearish momentum)
                    if bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1]:
                        position = 1
                        entry_price = prices['close'].iloc[i]
                        signals[i] = 0.25
                    elif bear_power_aligned[i] > 0 and bear_power_aligned[i] > bear_power_aligned[i-1]:
                        position = -1
                        entry_price = prices['close'].iloc[i]
                        signals[i] = -0.25
                elif adx_aligned[i] < 20:  # Ranging regime
                    # Long: Bear Power negative and turning up (oversold bounce)
                    # Short: Bull Power negative and turning down (overbought rejection)
                    if bear_power_aligned[i] < 0 and bear_power_aligned[i] > bear_power_aligned[i-1]:
                        position = 1
                        entry_price = prices['close'].iloc[i]
                        signals[i] = 0.25
                    elif bull_power_aligned[i] < 0 and bull_power_aligned[i] < bull_power_aligned[i-1]:
                        position = -1
                        entry_price = prices['close'].iloc[i]
                        signals[i] = -0.25
    
    return signals