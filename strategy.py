#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_Volume_Spike_ChopRegime
Hypothesis: Daily Camarilla R1/S1 breakout with volume spike (>2x 20-period volume MA) and chop regime filter (Choppiness Index > 61.8 = ranging market). 
In ranging markets, fade extreme moves at R1/S1 levels with mean-reversion logic. 
In trending markets (Chop < 38.2), follow breakouts. 
Uses 1d primary timeframe with 1w HTF for chop regime filter. 
Target 7-25 trades/year (30-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for chop regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Camarilla pivot points (R1, S1) from previous day ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r1 = pivot + (range_ * 1.1 / 4)  # R1 = pivot + range*(1.1/4)
    s1 = pivot - (range_ * 1.1 / 4)  # S1 = pivot - range*(1.1/4)
    
    # Align R1/S1 to 1d timeframe (wait for 1d bar close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1w Choppiness Index for regime filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr_period = 14
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of true ranges over period
    sum_tr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high_1w).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low_1w).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Choppiness Index = 100 * log10(sum_tr / (hh - ll)) / log10(period)
    # Avoid division by zero
    hl_range = hh - ll
    chop_raw = np.where(hl_range > 0, sum_tr / hl_range, np.nan)
    chop = 100 * np.log10(chop_raw) / np.log10(atr_period)
    
    # Align chop to 1d timeframe (wait for 1w bar close)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # === 1d Indicators (primary timeframe) ===
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume confirmation (>2x average)
        
        chop_val = chop_aligned[i]
        is_ranging = chop_val > 61.8  # chop > 61.8 = ranging market
        is_trending = chop_val < 38.2   # chop < 38.2 = trending market
        
        if position == 0:
            if is_ranging:
                # In ranging market: fade extreme moves at R1/S1
                # Long when price breaks below S1 (oversold bounce)
                if price < s1_aligned[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks above R1 (overbought reversal)
                elif price > r1_aligned[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
            else:
                # In trending market: follow breakouts
                # Long when price breaks above R1
                if price > r1_aligned[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks below S1
                elif price < s1_aligned[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price reverts to pivot or opposite extreme in ranging
            # or trend weakness in trending
            if is_ranging:
                if price > pivot[i]:  # revert to pivot
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In trending: exit on trend weakness (price < S1)
                if price < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverts to pivot or opposite extreme in ranging
            # or trend weakness in trending
            if is_ranging:
                if price < pivot[i]:  # revert to pivot
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In trending: exit on trend weakness (price > R1)
                if price > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_Volume_Spike_ChopRegime"
timeframe = "1d"
leverage = 1.0