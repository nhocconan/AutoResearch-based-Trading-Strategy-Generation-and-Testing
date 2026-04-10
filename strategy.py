#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (1d)
# - Regime: Bull if Bull Power > 0 and rising, Bear if Bear Power < 0 and falling
# - Entry: Long when Bull Power > 0 and rising + volume spike, Short when Bear Power < 0 and falling + volume spike
# - Volume confirmation: current 6h volume > 1.8x 20-period average
# - Designed for 6h timeframe: targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: regime filter adapts to higher timeframe momentum
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "6h_1d_elder_ray_regime_volume_v2"
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
    
    # Pre-compute 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Regime filter: Bull Power rising and Bear Power falling
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_falling = bear_power < np.roll(bear_power, 1)
    bull_power_rising[0] = False
    bear_power_falling[0] = False
    
    bull_regime = (bull_power > 0) & bull_power_rising
    bear_regime = (bear_power < 0) & bear_power_falling
    
    bull_regime_aligned = align_htf_to_ltf(prices, df_1d, bull_regime.astype(float))
    bear_regime_aligned = align_htf_to_ltf(prices, df_1d, bear_regime.astype(float))
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_regime_aligned[i]) or np.isnan(bear_regime_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: regime turns bearish or volume drops
            if bear_regime_aligned[i] > 0.5 or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: regime turns bullish or volume drops
            if bull_regime_aligned[i] > 0.5 or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for regime-aligned entries with volume confirmation
            if vol_spike[i]:
                if bull_regime_aligned[i] > 0.5:
                    position = 1
                    signals[i] = 0.25
                elif bear_regime_aligned[i] > 0.5:
                    position = -1
                    signals[i] = -0.25
    
    return signals