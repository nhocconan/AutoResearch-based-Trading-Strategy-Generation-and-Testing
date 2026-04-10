#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and ADX trend filter
# - Long: price breaks above Camarilla H3 (1d) + volume > 2.0x 20-period average + ADX(14) > 25
# - Short: price breaks below Camarilla L3 (1d) + volume > 2.0x 20-period average + ADX(14) > 25
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (2.5x ATR(14)) and time-based exit (max 3 bars hold)
# - Designed for 12h timeframe: targets 12-37 trades/year to avoid fee drag
# - Camarilla levels from 1d provide institutional support/resistance
# - Volume spike confirms institutional participation
# - ADX filter ensures we only trade in trending markets, avoiding chop
# - Works in bull/bear markets: breakouts capture momentum, ADX prevents false signals in ranging markets

name = "12h_1d_camarilla_pivot_breakout_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # L4 = close - 1.5*(high-low), L3 = close - 1.0*(high-low)
    rng = high_1d - low_1d
    camarilla_h3 = close_1d + 1.0 * rng  # H3 level
    camarilla_l3 = close_1d - 1.0 * rng  # L3 level
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (2.0 * avg_volume_20)  # Require strong volume spike
    
    # Pre-compute 12h ADX(14) for trend filter
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
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
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 12h ATR(14) for stoploss
    atr_14 = tr_14  # Already computed above
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    bars_held = 0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(adx[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            bars_held = 0
            continue
        
        if position == 1:  # Long position
            bars_held += 1
            # Exit: price breaks below Camarilla L3 OR stoploss hit OR max hold time reached
            if (low_12h[i] < camarilla_l3_aligned[i] or 
                close_12h[i] < entry_price - 2.5 * atr_14[i] or 
                bars_held >= 3):
                position = 0
                signals[i] = 0.0
                bars_held = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_held += 1
            # Exit: price breaks above Camarilla H3 OR stoploss hit OR max hold time reached
            if (high_12h[i] > camarilla_h3_aligned[i] or 
                close_12h[i] > entry_price + 2.5 * atr_14[i] or 
                bars_held >= 3):
                position = 0
                signals[i] = 0.0
                bars_held = 0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume and trend filters
            if vol_spike[i] and adx[i] > 25:  # Require volume spike and trending market
                # Long: price breaks above Camarilla H3
                if high_12h[i] > camarilla_h3_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                    bars_held = 1
                # Short: price breaks below Camarilla L3
                elif low_12h[i] < camarilla_l3_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
                    bars_held = 1
    
    return signals