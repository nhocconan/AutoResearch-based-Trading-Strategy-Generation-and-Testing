#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Elder Ray: Bull Power = Close - EMA13, Bear Power = EMA13 - Low
# - 1d ADX > 25 for trending market, < 20 for ranging
# - In trending (ADX>25): go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# - In ranging (ADX<20): fade extreme Elder Ray values (long when Bear Power < -std, short when Bull Power < -std)
# - Volume confirmation: current 6h volume > 1.5x 20-period average
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear/ranging markets: ADX regime adaptation

name = "6h_1d_elder_ray_adx_volume_v1"
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
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
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
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_6h = prices['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h Elder Ray components
    bull_power = close_6h - ema_13  # Bull Power = Close - EMA13
    bear_power = ema_13 - prices['low'].values  # Bear Power = EMA13 - Low
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    # Pre-compute Elder Ray volatility for ranging regime signals
    bull_power_std = pd.Series(bull_power).rolling(window=50, min_periods=50).std().values
    bear_power_std = pd.Series(bear_power).rolling(window=50, min_periods=50).std().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_spike[i]) or np.isnan(bull_power_std[i]) or np.isnan(bear_power_std[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            if adx_aligned[i] > 25:  # Trending regime
                # Exit long when Bull Power turns negative or stops rising
                if bull_power[i] <= 0 or (i > 100 and bull_power[i] < bull_power[i-1]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit long when Bear Power normalizes
                if bear_power[i] > -0.5 * bear_power_std[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx_aligned[i] > 25:  # Trending regime
                # Exit short when Bear Power turns negative or stops rising
                if bear_power[i] <= 0 or (i > 100 and bear_power[i] < bear_power[i-1]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit short when Bull Power normalizes
                if bull_power[i] > -0.5 * bull_power_std[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Look for entry signals based on regime
            if vol_spike[i]:
                if adx_aligned[i] > 25:  # Trending regime - momentum entries
                    # Long: Bull Power positive and rising
                    if bull_power[i] > 0 and (i <= 100 or bull_power[i] > bull_power[i-1]):
                        position = 1
                        signals[i] = 0.25
                    # Short: Bear Power positive and rising
                    elif bear_power[i] > 0 and (i <= 100 or bear_power[i] > bear_power[i-1]):
                        position = -1
                        signals[i] = -0.25
                elif adx_aligned[i] < 20:  # Ranging regime - mean reversion entries
                    # Long: Bear Power extremely negative (oversold)
                    if bear_power[i] < -2.0 * bear_power_std[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: Bull Power extremely negative (overbought)
                    elif bull_power[i] < -2.0 * bull_power_std[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals