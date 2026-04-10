#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period volume SMA AND chop < 61.8 (trending regime)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period volume SMA AND chop < 61.8 (trending regime)
# - Exit: Price reverts to Camarilla Pivot point (midpoint)
# - Position sizing: 0.25 discrete level to minimize fee impact while maintaining edge
# - Target: 12-30 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Camarilla pivots for structure, volume for confirmation, chop filter for regime

name = "12h_1d_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # H4 = Pivot + 1.1*(H-L)/2, L4 = Pivot - 1.1*(H-L)/2
    # H3 = Pivot + 1.1*(H-L)/4, L3 = Pivot - 1.1*(H-L)/4
    # We use H3/L3 for breakout, Pivot for exit
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and range
    pivot_1d = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    
    # Camarilla H3 and L3 levels
    h3_1d = pivot_1d + 1.1 * range_1d / 4.0
    l3_1d = pivot_1d - 1.1 * range_1d / 4.0
    pivot_1d_level = pivot_1d  # for exit
    
    # Align HTF levels to LTF (12h)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_level)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate Choppiness Index (14-period) on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (n * (HHV - LLV))) / log10(n)
    # We simplify: CHOP < 61.8 = trending, CHOP > 61.8 = ranging
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    hh_14_1d = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll_14_1d = df_1d['low'].rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14_1d = hh_14_1d - ll_14_1d
    range_14_1d = np.where(range_14_1d == 0, 1e-10, range_14_1d)
    
    chop_1d = 100 * np.log10(pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values / 
                             (14 * range_14_1d)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Regime filter: chop < 61.8 = trending (favor breakouts)
    trending_regime = chop_1d_aligned < 61.8
    
    for i in range(30, n):  # Start after warmup for HTF indicators
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        # Get 1d volume for current 12h bar (each 1d bar = 2 12h bars)
        idx_1d = i // 2
        if idx_1d < len(volume_1d):
            vol_confirm = volume_1d[idx_1d] > 2.0 * volume_sma_20_1d_aligned[i]
        else:
            vol_confirm = False
        
        # Camarilla breakout signals
        breakout_up = close[i] > h3_1d_aligned[i]  # Break above H3
        breakout_down = close[i] < l3_1d_aligned[i]  # Break below L3
        
        if position == 0:  # Flat - look for entry
            # Only enter in trending regime with volume confirmation
            if breakout_up and vol_confirm and trending_regime[i]:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and trending_regime[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when price reverts to pivot point
            if close[i] >= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when price reverts to pivot point
            if close[i] <= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals