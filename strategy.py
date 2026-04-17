#!/usr/bin/env python3
"""
1h_RegimeAdaptive_CCI_Volume_v1
1h strategy with 4h/1d multi-timeframe filters for direction and regime.
- Uses 4h CCI(20) for momentum direction (long when CCI>0, short when CCI<0)
- Uses 1d ADX(14) to filter regime: ADX>25 for trending, ADX<20 for ranging
- Uses 1h CCI(14) for entry timing with volume confirmation
- Volume filter: current volume > 1.5x 20-period average
- Session filter: 08-20 UTC to avoid low-liquidity hours
- Position size: 0.20 (20% of capital)
- Target: 15-30 trades/year (60-120 over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1h CCI(14) for entry timing ===
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=14, min_periods=14).mean().values
    mad = pd.Series(tp).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_1h = (tp - sma_tp) / (0.015 * mad)
    
    # === 1h volume average (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 4h CCI(20) for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tp_4h = (high_4h + low_4h + close_4h) / 3.0
    sma_tp_4h = pd.Series(tp_4h).rolling(window=20, min_periods=20).mean().values
    mad_4h = pd.Series(tp_4h).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_4h = (tp_4h - sma_tp_4h) / (0.015 * mad_4h)
    cci_4h_aligned = align_htf_to_ltf(prices, df_4h, cci_4h)
    
    # === 1d ADX(14) for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    plus_dm_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm_1d = np.concatenate([[0], plus_dm_1d])
    minus_dm_1d = np.concatenate([[0], minus_dm_1d])
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    minus_di_1d = 100 * pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(cci_1h[i]) or 
            np.isnan(cci_4h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: use 1d ADX
        adx_val = adx_1d_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Entry logic: only enter when flat
        if position == 0:
            # Determine bias from 4h CCI
            cci_4h_val = cci_4h_aligned[i]
            long_bias = cci_4h_val > 0
            short_bias = cci_4h_val < 0
            
            # Long conditions: 4h CCI > 0 + 1h CCI > 50 + volume + (trending OR ranging)
            if (long_bias and 
                cci_1h[i] > 50 and 
                volume_filter and 
                (is_trending or is_ranging)):
                signals[i] = 0.20
                position = 1
                continue
            # Short conditions: 4h CCI < 0 + 1h CCI < -50 + volume + (trending OR ranging)
            elif (short_bias and 
                  cci_1h[i] < -50 and 
                  volume_filter and 
                  (is_trending or is_ranging)):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: 1h CCI < -10 OR 4h CCI < 0 (bias change)
            if (cci_1h[i] < -10 or 
                cci_4h_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 1h CCI > 10 OR 4h CCI > 0 (bias change)
            if (cci_1h[i] > 10 or 
                cci_4h_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RegimeAdaptive_CCI_Volume_v1"
timeframe = "1h"
leverage = 1.0