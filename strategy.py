#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume spike filter and 1w choppiness regime filter
# Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 20-period volume SMA AND 1w CHOP > 61.8 (ranging market for mean reversion to pivot)
# Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 20-period volume SMA AND 1w CHOP > 61.8
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn
# Target: 20-50 trades/year on BTC/ETH, works in ranging markets where price respects pivot levels

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data once before loop for choppiness filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 4h Indicator: Camarilla Pivot Levels (R1, S1) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12.0
    
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1w Indicator: Choppiness Index (14-period) for regime filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_1w = np.where(range_14 > 0, 
                       100 * np.log10(tr_sum_14 / range_14) / np.log10(14), 
                       50.0)  # Neutral when range is zero
    
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Need 20 for volume SMA, 14 for CHOP
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_series = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume SMA
        vol_threshold = vol_sma_20_1d_aligned[i] * 1.5
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # CHOP filter: ranging market (CHOP > 61.8) for mean reversion to pivot
        chop_ranging = chop_1w_aligned[i] > 61.8
        
        # === LONG CONDITIONS ===
        # Price breaks above Camarilla R1 AND volume confirmation AND ranging market
        if (close[i] > r1_4h_aligned[i]) and vol_confirm and chop_ranging:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below Camarilla S1 AND volume confirmation AND ranging market
        elif (close[i] < s1_4h_aligned[i]) and vol_confirm and chop_ranging:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1S1_1dVolSpike_1wCHOP_Filter_v1"
timeframe = "4h"
leverage = 1.0