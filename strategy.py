#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and ADX trend filter
# - Long when price breaks above Donchian(20) high with 1d volume spike and 1d ADX > 25
# - Short when price breaks below Donchian(20) low with 1d volume spike and 1d ADX > 25
# - Exit when price reverts to Donchian midpoint or ATR-based stoploss (2.0x ATR)
# - Discrete position sizing (0.25) to minimize fee churn
# - Targets 20-50 trades/year to avoid fee drag

name = "4h_1d_donchian_breakout_volume_adx_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX(14) for trend strength
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr3 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14)
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 4h Donchian(20) channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # 4h ATR(14) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(np.diff(high_4h, prepend=high_4h[0]))
    tr_4h3 = np.abs(np.diff(low_4h, prepend=low_4h[0]))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_14_4h = np.zeros_like(tr_4h)
    atr_14_4h[13] = np.mean(tr_4h[:14])
    for i in range(14, len(tr_4h)):
        atr_14_4h[i] = (atr_14_4h[i-1] * 13 + tr_4h[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_14_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] > donchian_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] < donchian_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume and trend filters
            if vol_spike_1d_aligned[i] and adx_14_aligned[i] > 25:
                # Long signal: price breaks above Donchian high
                if prices['high'].iloc[i] > donchian_high[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low
                elif prices['low'].iloc[i] < donchian_low[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h[i]
                    signals[i] = -0.25
    
    return signals