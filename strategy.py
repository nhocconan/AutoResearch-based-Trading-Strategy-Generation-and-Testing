#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V1
Hypothesis: 12h Camarilla pivot breakouts at R1/S1 levels with volume confirmation and 1d chop regime filter capture institutional order flow in both bull and bear markets. Uses ATR stoploss for risk control. Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. Works in ranging markets via mean reversion at pivot levels and in trending markets via breakouts with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla calculation and chop filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + (rng * 1.1 / 12)
    camarilla_s1 = close_1d - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d Chop Regime Filter (Ehler's Chopiness Index) ===
    # Chop = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high)-min(low)))) over period
    chop_period = 14
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=chop_period, min_periods=chop_period).mean().values
    
    max_high = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    
    sum_atr = pd.Series(atr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    range_max_min = max_high - min_low
    chop = 100 * np.log10(sum_atr / (np.log10(chop_period) * range_max_min))
    chop = np.where(range_max_min == 0, 50, chop)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Chop regime: > 61.8 = ranging (mean revert at pivots), < 38.2 = trending (breakout)
    # We use chop > 50 as ranging filter to avoid false breakouts in strong trends
    ranging_regime = chop_aligned > 50
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * vol_ma)
    
    # ATR (14-period) for stoploss
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(volume_spike[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike in ranging market (mean reversion setup)
            if price > camarilla_r1_aligned[i] and volume_spike[i] and ranging_regime[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below S1 with volume spike in ranging market
            elif price < camarilla_s1_aligned[i] and volume_spike[i] and ranging_regime[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price returns to mid-point between R1 and S1 or breaks below S1
            elif price < camarilla_s1_aligned[i] or price > (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price returns to mid-point or breaks above R1
            elif price > camarilla_r1_aligned[i] or price < (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V1"
timeframe = "12h"
leverage = 1.0