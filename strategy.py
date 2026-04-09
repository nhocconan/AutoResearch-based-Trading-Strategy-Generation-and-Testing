#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + 12h Supertrend trend filter + volume confirmation
# - Primary signal: 6h Elder Ray Bull/Bear Power crosses zero with volume confirmation
# - Trend filter: 12h Supertrend (ATR=10, mult=3.0) - only take longs in uptrend, shorts in downtrend
# - Volume confirmation: 6h volume > 20-period EMA of volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Elder Ray measures bull/bear power behind moves, Supertrend filter ensures
#   trades align with higher timeframe trend, reducing false signals during trend exhaustion

name = "6h_12h_elderray_supertrend_volume_v1"
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
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Supertrend for trend direction (ATR=10, mult=3.0)
    # Calculate ATR
    tr1 = pd.Series(high_12h).rolling(2).max() - pd.Series(low_12h).rolling(2).min()
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + (3.0 * atr_12h)
    lower_band_12h = hl2_12h - (3.0 * atr_12h)
    
    supertrend_12h = np.zeros_like(close_12h)
    uptrend_12h = np.ones_like(close_12h, dtype=bool)
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upper_band_12h[i-1]:
            uptrend_12h[i] = True
        elif close_12h[i] < lower_band_12h[i-1]:
            uptrend_12h[i] = False
        else:
            uptrend_12h[i] = uptrend_12h[i-1]
            if uptrend_12h[i] and lower_band_12h[i] < lower_band_12h[i-1]:
                lower_band_12h[i] = lower_band_12h[i-1]
            if not uptrend_12h[i] and upper_band_12h[i] > upper_band_12h[i-1]:
                upper_band_12h[i] = upper_band_12h[i-1]
        
        supertrend_12h[i] = lower_band_12h[i] if uptrend_12h[i] else upper_band_12h[i]
    
    # Align 12h Supertrend uptrend to 6h timeframe (completed 12h bar only)
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Elder Ray Index: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 6h volume regime: volume > 20-period EMA of volume
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_regime = volume > volume_ema_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(uptrend_12h_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power crosses above zero (bulls losing control) OR Supertrend turns down
            if bear_power[i] > 0 or uptrend_12h_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power crosses above zero (bears losing control) OR Supertrend turns up
            if bull_power[i] > 0 or uptrend_12h_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray extremes with volume confirmation and Supertrend filter
            # Long: Bull Power crosses above zero AND volume regime AND Supertrend uptrend
            if bull_power[i] > 0 and bull_power[i-1] <= 0 and volume_regime[i] and uptrend_12h_aligned[i] > 0.5:
                position = 1
                signals[i] = 0.25
            # Short: Bear Power crosses above zero AND volume regime AND Supertrend downtrend
            elif bear_power[i] > 0 and bear_power[i-1] <= 0 and volume_regime[i] and uptrend_12h_aligned[i] < 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals