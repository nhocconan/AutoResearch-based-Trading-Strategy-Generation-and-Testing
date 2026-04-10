#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) from 1d timeframe
# - Regime filter: 1d ADX(14) > 25 for trending markets, < 20 for ranging markets
# - In trending regime (ADX>25): trade breakouts when Bull/Bear Power > 0 and volume > 1.5x average
# - In ranging regime (ADX<20): trade mean reversion when Bull/Bear Power < 0 and price near EMA
# - Volume confirmation: current 6h volume > 1.5x 20-period average to confirm participation
# - Designed for 6h timeframe: targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: regime filter adapts strategy to market conditions
# - Uses discrete position sizing (0.25) to minimize fee churn

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
    
    # Pre-compute 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
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
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema_13_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive (bullish momentum fading) or price closes below EMA
            if bear_power_aligned[i] > 0 or prices['close'].iloc[i] < ema_13_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative (bearish momentum fading) or price closes above EMA
            if bull_power_aligned[i] < 0 or prices['close'].iloc[i] > ema_13_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Regime-based entry logic
            if vol_spike[i]:
                if adx_aligned[i] > 25:  # Trending regime
                    # Breakout long: Bull Power > 0 and rising
                    if bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Breakout short: Bear Power < 0 and falling
                    elif bear_power_aligned[i] < 0 and bear_power_aligned[i] < bear_power_aligned[i-1]:
                        position = -1
                        signals[i] = -0.25
                elif adx_aligned[i] < 20:  # Ranging regime
                    # Mean reversion long: Bear Power < 0 and price near EMA (oversold)
                    if bear_power_aligned[i] < 0 and prices['close'].iloc[i] < ema_13_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Mean reversion short: Bull Power > 0 and price near EMA (overbought)
                    elif bull_power_aligned[i] > 0 and prices['close'].iloc[i] > ema_13_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals