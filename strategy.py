#!/usr/bin/env python3
"""
12h_KAMA_Direction_RegimeFilter_VolumeSpike
Hypothesis: 12h KAMA trend direction filtered by 1d choppiness regime and volume spike (2.0x average).
Long when KAMA trending up, CHOP > 61.8 (range), and volume confirmed. Short when KAMA trending down, CHOP > 61.8, and volume confirmed.
Uses 1d HTF for chop regime to avoid whipsaw in trending markets and capture mean reversion in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
Works in both bull and bear markets by adapting to regime: mean revert in range, follow trend in strong trends (though chop filter favors range).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h KAMA for trend direction ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change_12h = np.abs(np.diff(close_12h, n=10))
    volatility_12h = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)
    er_12h = np.divide(change_12h, volatility_12h, out=np.zeros_like(change_12h), where=volatility_12h!=0)
    # Smooth ER with smoothing constants
    sc_12h = (er_12h * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # Initialize KAMA
    kama_12h = np.full_like(close_12h, np.nan)
    kama_12h[30] = close_12h[30]  # start after 30 periods
    for i in range(31, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    # Align 12h KAMA to 4h timeframe
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.diff(kama_12h_aligned, prepend=kama_12h_aligned[0])
    kama_dir = np.where(kama_dir > 0, 1, np.where(kama_dir < 0, -1, 0))
    
    # === 1d OHLC for Choppiness Index calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) for chop
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate max/min close over 14 periods
    max_close_1d = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close_1d = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(atr14) / (max_close - min_close)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    denominator = max_close_1d - min_close_1d
    chop_1d = 100 * np.log10(sum_atr_14 / denominator) / np.log10(14)
    # Handle division by zero or invalid values
    chop_1d = np.where((denominator > 0) & (sum_atr_14 > 0), chop_1d, 50.0)
    # Align 1d chop to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_dir[i]) or np.isnan(chop_1d_aligned[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        kama_direction = kama_dir[i]
        chop = chop_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume_now > 2.0 * vol_avg
        # Regime filter: chop > 61.8 indicates ranging market (mean reversion favorable)
        ranging_market = chop > 61.8
        
        if position == 0:
            # Only enter in ranging markets with volume confirmation
            # Long when KAMA trending up, short when KAMA trending down
            long_condition = ranging_market and (kama_direction == 1) and volume_confirmed
            short_condition = ranging_market and (kama_direction == -1) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if market becomes trending (chop < 38.2) or KAMA reverses
            elif (chop < 38.2) or (kama_direction == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if market becomes trending (chop < 38.2) or KAMA reverses
            elif (chop < 38.2) or (kama_direction == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_RegimeFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0