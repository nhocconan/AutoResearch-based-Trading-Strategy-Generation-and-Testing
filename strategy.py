#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike filter and ADX regime confirmation.
- Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND ADX > 25 (trending market)
- Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND ADX > 25 (trending market)
- Exit on opposite Camarilla breakout (L3 for long exit, H3 for short exit) or ADX < 20 (range market)
- Fixed position size of 0.25 to balance return and drawdown
- Uses 12h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Camarilla levels provide institutional support/resistance; volume spike confirms participation; ADX filters chop
- Designed to work in both bull (trend continuation) and bear (trend continuation) markets via ADX filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 12h bar
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # Using previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # first bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    H3 = prev_close + 1.1 * camarilla_range / 2
    L3 = prev_close - 1.1 * camarilla_range / 2
    
    # Get 1d data ONCE before loop for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / (vol_ma + 1e-10)
    vol_spike = vol_ratio > 1.5
    
    # Calculate 1d ADX(14) for regime filter
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']),
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)),
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    # Handle first NaN values
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = np.nan  # first bar has no previous close
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Regime: trending if ADX > 25, ranging if ADX < 20
    trending_regime = adx_aligned > 25
    ranging_regime = adx_aligned < 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 14) + 1  # volume MA, ADX calculation, and prev bar shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above H3 AND volume spike AND trending regime
            if close[i] > H3[i] and vol_spike_aligned[i] and trending_regime[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below L3 AND volume spike AND trending regime
            elif close[i] < L3[i] and vol_spike_aligned[i] and trending_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below L3 OR ADX drops below 20 (range market)
            if close[i] < L3[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above H3 OR ADX drops below 20 (range market)
            if close[i] > H3[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dVolumeSpike_ADXRegime_v1"
timeframe = "12h"
leverage = 1.0