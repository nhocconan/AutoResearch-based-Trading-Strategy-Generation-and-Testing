#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakouts with volume confirmation and ATR-based position sizing.
# Donchian(20) from 1d provides major support/resistance levels that work in both bull and bear markets.
# Volume confirmation (current 4h volume > 1.8x 20-period average) filters false breakouts.
# Position size scales with inverse volatility (0.02/ATR) to maintain consistent risk, clamped to 0.20-0.30.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4h_1d_donchian_volume_atr_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d ATR (14-period) for volatility filtering and position sizing
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian levels and ATR to 4h timeframe
    dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    dm_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or
            np.isnan(dm_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(vol_ma_20[i]) or atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.8x average 4h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Dynamic position size: inverse volatility scaling (target ~0.25 at median ATR)
        # Clamp ATR to reasonable range to avoid extreme position sizes
        atr_clamped = np.clip(atr_aligned[i], 0.001, 0.10)  # Avoid division by zero or tiny ATR
        base_size = 0.25
        vol_scaling = 0.02 / atr_clamped  # Scale so 2% ATR gives ~0.25 size
        vol_scaling = np.clip(vol_scaling, 0.5, 2.0)  # Clamp scaling to reasonable range
        position_size = base_size * vol_scaling
        position_size = np.clip(position_size, 0.20, 0.30)  # Final clamp to 0.20-0.30
        
        if position == 1:  # Long position
            # Exit on retracement to Donchian midpoint
            if close[i] < dm_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to Donchian midpoint
            if close[i] > dm_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout trading with volume confirmation
            # Long on Donchian high breakout, Short on Donchian low breakout
            if close[i] > dh_aligned[i]:
                position = 1
                signals[i] = position_size
            elif close[i] < dl_aligned[i]:
                position = -1
                signals[i] = -position_size
    
    return signals