#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and 1w ADX trend filter
# - Williams %R(14) identifies overbought/oversold conditions for mean reversion
# - 1w ADX(14) > 25 ensures we trade with the higher timeframe trend (works in bull/bear)
# - 1d volume spike (>2.0x 20-period average) confirms institutional participation
# - Designed for 4h timeframe: targets 30-60 trades/year (120-240 total over 4 years)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "4h_1d_1w_williamsr_adx_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
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
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 1d volume confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 4h Williams %R(14)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or Williams %R exits oversold territory
            if prices['close'].iloc[i] < entry_price - 2.0 * atr_14[i] or williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or Williams %R exits overbought territory
            if prices['close'].iloc[i] > entry_price + 2.0 * atr_14[i] or williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with trend and volume filters
            if vol_spike_1d_aligned[i] and adx_aligned[i] > 25:
                # Mean reversion long: Williams %R oversold (< -80)
                if williams_r[i] < -80:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Mean reversion short: Williams %R overbought (> -20)
                elif williams_r[i] > -20:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals