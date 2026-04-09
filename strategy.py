#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# - Uses 12h Donchian channel breakout for trend following entries
# - Trend filter: 1w ADX(14) > 25 to ensure trading only in trending markets (works in bull/bear)
# - Volume confirmation: 12h volume > 2.0x 20-period average to ensure breakout strength
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Novelty: Combines Donchian breakout with 1w ADX regime filter to avoid false signals in ranging markets
# - ADX filter prevents whipsaws during consolidation, works in both bull and bear trends

name = "12h_1w_donchian_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w indicators for ADX trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14) for trend regime filter
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = tr_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move_1w = high_1w - np.roll(high_1w, 1)
    down_move_1w = np.roll(low_1w, 1) - low_1w
    plus_dm_1w = np.where((up_move_1w > down_move_1w) & (up_move_1w > 0), up_move_1w, 0.0)
    minus_dm_1w = np.where((down_move_1w > up_move_1w) & (down_move_1w > 0), down_move_1w, 0.0)
    
    # Smoothed +DM, -DM, and TR
    plus_dm_1w_smooth = pd.Series(plus_dm_1w).rolling(window=14, min_periods=14).mean().values
    minus_dm_1w_smooth = pd.Series(minus_dm_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_smooth = pd.Series(atr_1w).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di_1w = 100 * plus_dm_1w_smooth / atr_1w_smooth
    minus_di_1w = 100 * minus_dm_1w_smooth / atr_1w_smooth
    
    # DX and ADX
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    
    # Trend regime: ADX > 25 indicates trending market
    trending_regime = adx_1w > 25.0
    
    # Align 1w trend regime to 12h timeframe (completed 1w bar only)
    trending_regime_aligned = align_htf_to_ltf(prices, df_1w, trending_regime)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # 12h ATR(14) for trailing stop
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
            np.isnan(trending_regime_aligned[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and trend filter
            # Long: price breaks above Donchian high AND volume spike AND trending regime
            if high[i] >= donchian_high[i] and volume_spike[i] and trending_regime_aligned[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND trending regime
            elif low[i] <= donchian_low[i] and volume_spike[i] and trending_regime_aligned[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals