#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Camarilla levels from 12h: R3/S3 for mean reversion fade, R4/S4 for breakout continuation
# - 12h ADX(14) > 20 to ensure sufficient trend strength and avoid choppy markets
# - Volume confirmation: current 6h volume > 2.0x 20-period average for breakout validation
# - Discrete position sizing (0.25) to minimize fee churn
# - Designed for 6h timeframe: targets 12-30 trades/year to avoid fee drag
# - Works in bull/bear markets: ADX filter ensures we trade with higher timeframe trend bias

name = "6h_12h_camarilla_adx_volume_v1"
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
    
    # Pre-compute 12h ADX(14) for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
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
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Need at least 2 periods of 12h data for Camarilla calculation
            if i < 12:  # Not enough 12h bars yet (each 12h bar = 24 of 6h bars)
                signals[i] = 0.0
                continue
                
            # Get the most recent completed 12h bar index
            idx_12h = (i // 24) - 1  # -1 ensures we use only completed 12h bars
            if idx_12h < 0:
                signals[i] = 0.0
                continue
                
            # Calculate Camarilla levels from previous 12h bar
            h = high_12h[idx_12h]
            l = low_12h[idx_12h]
            c = close_12h[idx_12h]
            rng = h - l
            
            if rng <= 0:  # Avoid division by zero
                signals[i] = 0.0
                continue
                
            # Camarilla levels
            r4 = c + rng * 1.1 / 2
            r3 = c + rng * 1.1 / 4
            s3 = c - rng * 1.1 / 4
            s4 = c - rng * 1.1 / 2
            
            price = prices['close'].iloc[i]
            
            # Look for breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 20:
                # Breakout long: price closes above R4
                if price > r4:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price closes below S4
                elif price < s4:
                    position = -1
                    signals[i] = -0.25
        else:  # In position - look for exit
            # Exit conditions: re-entry into Camarilla H-L range or opposite Camarilla level
            if i < 12:  # Not enough 12h bars yet
                signals[i] = 0.25 if position == 1 else -0.25
                continue
                
            # Get the most recent completed 12h bar index
            idx_12h = (i // 24) - 1
            if idx_12h < 0:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
                
            # Calculate Camarilla levels from previous 12h bar
            h = high_12h[idx_12h]
            l = low_12h[idx_12h]
            c = close_12h[idx_12h]
            rng = h - l
            
            if rng <= 0:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
                
            # Camarilla levels
            r3 = c + rng * 1.1 / 4
            s3 = c - rng * 1.1 / 4
            
            price = prices['close'].iloc[i]
            
            if position == 1:  # Long position
                # Exit: price re-enters H-L range or breaks below S3 (failed breakout)
                if price <= h and price >= l or price < s3:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Short position
                # Exit: price re-enters H-L range or breaks above R3 (failed breakout)
                if price <= h and price >= l or price > r3:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals