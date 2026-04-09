#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike
# - Uses 6h Williams %R(14) for oversold/overbought signals (long < -80, short > -20)
# - Filters by 1d ADX(14) > 25 to ensure trading only in trending markets
# - Requires 1d volume > 1.5x 20-period average for institutional confirmation
# - Exits when Williams %R reverts to -50 (mean reversion target)
# - Position size: 0.25 (25% of capital) for conservative risk management
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to minimize fee drag
# - Williams %R captures extreme sentiment reversals that work in both bull and bear markets
# - ADX filter ensures we only trade when there's a clear trend to revert from
# - Volume spike confirms participation from smart money

name = "6h_1d_williamsr_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ADX calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ADX(14) calculation
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_di_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * plus_di_ma / tr_ma
    minus_di = 100 * minus_di_ma / tr_ma
    dx = np.where((plus_di + minus_di) > 0,
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    
    # Align all 1d indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(williams_r[i]) or adx_aligned[i] < 25):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R reverts to -50 (mean reversion)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R reverts to -50 (mean reversion)
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume confirmation and ADX filter
            if williams_r[i] < -80 and volume_spike_aligned[i]:  # Oversold + volume
                position = 1
                signals[i] = 0.25
            elif williams_r[i] > -20 and volume_spike_aligned[i]:  # Overbought + volume
                position = -1
                signals[i] = -0.25
    
    return signals