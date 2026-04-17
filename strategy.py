#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_Regime_v1
Based on Camarilla pivot levels from daily timeframe.
Long when price breaks above R1 with volume spike and chop regime > 61.8 (range).
Short when price breaks below S1 with volume spike and chop regime > 61.8 (range).
Exit when price returns to pivot point (close).
Uses 1w EMA200 for higher timeframe trend filter (only trade in direction of trend).
Designed to capture mean reversion in ranging markets with trend filter.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily OHLC for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    H_1d = df_1d['high'].values
    L_1d = df_1d['low'].values
    C_1d = df_1d['close'].values
    
    R1 = C_1d + ((H_1d - L_1d) * 1.0833)
    S1 = C_1d - ((H_1d - L_1d) * 1.0833)
    PP = (H_1d + L_1d + C_1d) / 3.0
    
    # Align daily levels to 12h timeframe (wait for daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # === Volume spike detection (volume > 2x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Chopiness Index (range detection) ===
    # CHOP = 100 * log10(sum(TR over n) / (n * (HHV - LLV))) / log10(n)
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market
    n_chop = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n_chop, min_periods=n_chop).sum().values
    hh = pd.Series(high).rolling(window=n_chop, min_periods=n_chop).max().values
    ll = pd.Series(low).rolling(window=n_chop, min_periods=n_chop).min().values
    chop = 100 * np.log10(atr_sum / (n_chop * (hh - ll) + 1e-10)) / np.log10(n_chop)
    chop[~np.isfinite(chop)] = 50  # default to neutral when undefined
    
    chop_range = chop > 61.8  # ranging market
    
    # === 1w EMA200 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = max(100, 200)  # ensure all indicators are ready
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1, volume spike, chop > 61.8 (range), price above 1w EMA200
            if (close[i] > R1_aligned[i] and 
                volume_spike[i] and 
                chop_range[i] and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1, volume spike, chop > 61.8 (range), price below 1w EMA200
            elif (close[i] < S1_aligned[i] and 
                  volume_spike[i] and 
                  chop_range[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to pivot point (PP)
        elif position == 1:
            # Exit long: price crosses below PP
            if close[i] < PP_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above PP
            if close[i] > PP_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0