#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V2
Hypothesis: Camarilla R1/S1 levels on 1d provide institutional support/resistance. 
Breakout with volume confirmation and choppiness regime filter captures institutional flow.
ATR-based stoploss manages risk. Works in bull/bear: regime filter adapts to market state.
Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for choppiness regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w Choppiness Index for regime filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_tr / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.where(range_hl > 0, 100 * np.log10(sum_tr / range_hl) / np.log10(14), 50)
    chop = np.nan_to_num(chop, nan=50.0)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # === 1d Camarilla Pivot Levels (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Camarilla levels: R1, R2, R3, R4 and S1, S2, S3, S4
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    rang = prev_high - prev_low
    R1 = prev_close + (rang * 1.1 / 12)
    R2 = prev_close + (rang * 1.1 / 6)
    R3 = prev_close + (rang * 1.1 / 4)
    R4 = prev_close + (rang * 1.1 / 2)
    S1 = prev_close - (rang * 1.1 / 12)
    S2 = prev_close - (rang * 1.1 / 6)
    S3 = prev_close - (rang * 1.1 / 4)
    S4 = prev_close - (rang * 1.1 / 2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR (10-period) for stoploss
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_d = np.maximum(np.maximum(tr1_d, tr2_d), tr3_d)
    atr = pd.Series(tr_d).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(vol_ratio[i]) 
            or np.isnan(atr[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol_ok = vol_ratio[i] > 1.5
        
        # Regime filter: CHOP > 50 = ranging (mean revert), CHOP < 50 = trending (breakout)
        # We want breakouts in trending markets, mean reversion in ranging markets
        # For this strategy, we focus on breakouts in trending markets (CHOP < 50)
        trending_regime = chop_aligned[i] < 50
        
        if position == 0:
            # Long breakout above R1 with volume and trending regime
            if price > R1[i] and vol_ok and trending_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakdown below S1 with volume and trending regime
            elif price < S1[i] and vol_ok and trending_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price reaches R2 (take profit) or reverses below R1
            elif price >= R2[i] or price < R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price reaches S2 (take profit) or reverses above S1
            elif price <= S2[i] or price > S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V2"
timeframe = "1d"
leverage = 1.0