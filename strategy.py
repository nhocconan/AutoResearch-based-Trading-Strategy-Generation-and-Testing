#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (from 1d)
# - Regime filter: ADX(14) from 1d > 25 to ensure trending markets, < 20 for ranging
# - Volume confirmation: current 6h volume > 1.8x 20-period average
# - In trending regime (ADX>25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# - In ranging regime (ADX<20): fade extremes - long when Bull Power < -0.5*ATR, short when Bear Power > 0.5*ATR
# - Designed for 6h timeframe: targets 12-35 trades/year (50-140 total over 4 years)
# - Works in bull/bear markets: regime adaptation avoids whipsaws in ranging markets

name = "6h_1d_elder_ray_regime_volume_v1"
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
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_1d - ema13  # Bull Power
    bear_power = low_1d - ema13   # Bear Power
    
    # ATR for volatility normalization
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX for regime detection
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            if adx_aligned[i] > 25:  # Trending regime
                # Exit when Bull Power turns negative or volume drops
                if bull_power_aligned[i] <= 0 or not vol_spike[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when price reverts to mean (Bull Power > -0.2*ATR)
                if bull_power_aligned[i] > -0.2 * atr_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx_aligned[i] > 25:  # Trending regime
                # Exit when Bear Power turns positive or volume drops
                if bear_power_aligned[i] >= 0 or not vol_spike[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when price reverts to mean (Bear Power < 0.2*ATR)
                if bear_power_aligned[i] < 0.2 * atr_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if vol_spike[i]:
                if adx_aligned[i] > 25:  # Trending regime - follow momentum
                    # Long: Bull Power positive and rising
                    if bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short: Bear Power negative and falling
                    elif bear_power_aligned[i] < 0 and bear_power_aligned[i] < bear_power_aligned[i-1]:
                        position = -1
                        signals[i] = -0.25
                elif adx_aligned[i] < 20:  # Ranging regime - mean reversion
                    # Long: Bull Power significantly negative (oversold)
                    if bull_power_aligned[i] < -0.5 * atr_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: Bear Power significantly positive (overbought)
                    elif bear_power_aligned[i] > 0.5 * atr_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals