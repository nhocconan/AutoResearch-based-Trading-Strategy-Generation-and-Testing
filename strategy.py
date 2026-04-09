#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ATR volatility filter and volume confirmation
# - Uses 4h Donchian channel breakout for trend following entries
# - Volatility filter: Only trade when 12h ATR(14) > 20-period median ATR (avoid low volatility chop)
# - Volume confirmation: 4h volume > 1.8x 20-period average to ensure breakout strength
# - ATR(14) trailing stop at 2.5x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in both bull/bear: Donchian captures trends, volatility filter prevents whipsaws in ranging markets
# - Multi-timeframe: 12h volatility filter provides regime context without look-ahead

name = "4h_12h_donchian_vol_volume_v1"
timeframe = "4h"
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
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    # Volatility regime: ATR > 20-period median ATR (avoid low volatility chop)
    median_atr_20 = pd.Series(atr_12h).rolling(window=20, min_periods=20).median().values
    vol_regime = atr_12h > median_atr_20
    
    # Align 12h volatility regime to 4h timeframe (completed 12h bar only)
    vol_regime_aligned = align_htf_to_ltf(prices, df_12h, vol_regime)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume > 1.8x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    # 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_regime_aligned[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high
            if low[i] <= highest_since_entry - (2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low
            if high[i] >= lowest_since_entry + (2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and volatility filter
            # Long: price breaks above Donchian high AND volume spike AND high volatility regime
            if high[i] >= donchian_high[i] and volume_spike[i] and vol_regime_aligned[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND high volatility regime
            elif low[i] <= donchian_low[i] and volume_spike[i] and vol_regime_aligned[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals