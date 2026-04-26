#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Regime_v3
Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter, volume confirmation, and choppiness regime filter.
- Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Long when price breaks above R1 AND 1d uptrend AND volume spike AND choppiness < 61.8 (trending)
- Short when price breaks below S1 AND 1d downtrend AND volume spike AND choppiness < 61.8 (trending)
- Camarilla levels from 1d provide institutional support/resistance
- 1d EMA34 trend filter reduces whipsaw in bear markets
- Volume spike (2.0x 20-period average) confirms institutional participation
- Choppiness regime filter (CHOP < 61.8) avoids ranging markets where breakouts fail
- Designed for low frequency with proven edge on BTC/ETH from Camarilla's accuracy
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels, EMA34 trend, and choppiness
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels (R1, S1)
    # Camarilla: R1 = close + 1.1/12 * (high - low), S1 = close - 1.1/12 * (high - low)
    camarilla_range = df_1d['high'] - df_1d['low']
    r1_1d = df_1d['close'] + (1.1 / 12) * camarilla_range
    s1_1d = df_1d['close'] - (1.1 / 12) * camarilla_range
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for support/resistance levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d.values)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d.values)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_14 = []
    tr = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr, df_1d['low'].values)
    tr = np.maximum(tr, np.roll(tr, 1))  # True Range: max(high-low, high-prev_close, low-prev_close)
    tr[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # First TR
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    
    chop_raw = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)  # Replace NaN with neutral value
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Calculate volume spike (20-period volume average on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 14 for ATR, 20 for volume MA)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation, trend filter, and regime filter
        if position == 0:
            # Long: Price breaks above R1 AND 1d uptrend AND volume spike AND trending regime (CHOP < 61.8)
            if close[i] > r1_1d_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i] and chop_1d_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 AND 1d downtrend AND volume spike AND trending regime (CHOP < 61.8)
            elif close[i] < s1_1d_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i] and chop_1d_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below S1 OR 1d trend turns down OR choppiness becomes too high (ranging)
            if close[i] < s1_1d_aligned[i] or close[i] < ema34_1d_aligned[i] or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above R1 OR 1d trend turns up OR choppiness becomes too high (ranging)
            if close[i] > r1_1d_aligned[i] or close[i] > ema34_1d_aligned[i] or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Regime_v3"
timeframe = "12h"
leverage = 1.0