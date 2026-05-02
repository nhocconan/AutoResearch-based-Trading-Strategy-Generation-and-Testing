#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme + 1d ADX Regime + Volume Spike
# Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Williams %R identifies overbought/oversold conditions with mean reversion edge
# 1d ADX > 25 filters for trending markets (avoid chop) and < 20 for ranging
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing: 0.25 balances exposure and risk
# Works in bull via trend continuation and bear via mean reversion in ranging markets

name = "4h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14) with proper handling
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high_1d.diff()
    dm_minus = low_1d.diff().abs() * -1  # Invert to get positive values for down moves
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smooth TR and DM
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False).mean()
    
    # DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / atr)
    di_minus = 100 * (dm_minus_smooth / atr)
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    
    # Align ADX to 4h timeframe
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 4h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Calculate 4h volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 30)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Regime filter: ADX > 25 = trending (use breakout logic), ADX < 20 = ranging (use mean reversion)
            if adx_aligned[i] > 25:
                # Trending regime: Donchian-like breakout with Williams %R confirmation
                # Long: price breaks above recent high AND Williams %R > -20 (not overbought) AND volume spike
                # Short: price breaks below recent low AND Williams %R < -80 (not oversold) AND volume spike
                recent_high = pd.Series(high).rolling(window=10, min_periods=10).max().shift(1).values[i]
                recent_low = pd.Series(low).rolling(window=10, min_periods=10).min().shift(1).values[i]
                
                if (close[i] > recent_high and 
                    williams_r[i] > -20 and 
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] < recent_low and 
                      williams_r[i] < -80 and 
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif adx_aligned[i] < 20:
                # Ranging regime: Williams %R mean reversion
                # Long: Williams %R < -80 (oversold) AND volume spike
                # Short: Williams %R > -20 (overbought) AND volume spike
                if (williams_r[i] < -80 and 
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                elif (williams_r[i] > -20 and 
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX between 20-25): no trading
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR ADX < 20 (trend weakening)
            if williams_r[i] > -20 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR ADX < 20 (trend weakening)
            if williams_r[i] < -80 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals