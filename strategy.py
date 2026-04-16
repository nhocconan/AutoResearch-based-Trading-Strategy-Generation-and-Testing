# 12h_Camarilla_R1S1_Breakout_Volume_Spike
# Hypothesis: At 12h timeframe, Camarilla pivot levels (R1/S1) act as strong support/resistance.
# A breakout above R1 or below S1 with volume spike (>2x 20-period average) indicates momentum.
# In ranging markets (Choppiness Index > 61.8), we mean-revert at H5/L5 levels.
# This strategy works in both bull/bear regimes by adapting to market structure.
# Target: 15-30 trades/year, low frequency to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # H5 = C + (H - L) * 1.1 / 2
    # L5 = C - (H - L) * 1.1 / 2
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r1_12h = close_12h + range_12h * 1.1 / 12
    s1_12h = close_12h - range_12h * 1.1 / 12
    h5_12h = close_12h + range_12h * 1.1 / 2
    l5_12h = close_12h - range_12h * 1.1 / 2
    
    # Align Camarilla levels to lower timeframe (1h for precision)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    h5_12h_aligned = align_htf_to_ltf(prices, df_12h, h5_12h)
    l5_12h_aligned = align_htf_to_ltf(prices, df_12h, l5_12h)
    
    # === 1d data (HTF for regime detection) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period) for regime detection
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(sum(TR)/ATR) / log10(14)
    # Higher values (>61.8) indicate ranging/choppy market
    # Lower values (<38.2) indicate trending market
    choppiness = 100 * np.log10(tr_sum_14 / (atr_1d * 14)) / np.log10(14)
    choppiness_aligned = align_htf_to_ltf(prices, df_1d, choppiness)
    
    # === 1h data (for entry timing precision) ===
    df_1h = get_htf_data(prices, '1h')
    volume_1h = df_1h['volume'].values
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1h / (vol_ma_20 + 1e-10)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1h, vol_ratio)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Position tracking
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or
            np.isnan(h5_12h_aligned[i]) or np.isnan(l5_12h_aligned[i]) or
            np.isnan(choppiness_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        r1 = r1_12h_aligned[i]
        s1 = s1_12h_aligned[i]
        h5 = h5_12h_aligned[i]
        l5 = l5_12h_aligned[i]
        chop = choppiness_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price reaches H5 (take profit) or closes below S1 (stop)
            if price >= h5 or price < s1:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price reaches L5 (take profit) or closes above R1 (stop)
            if price <= l5 or price > r1:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and in_session:
            # Determine market regime using Choppiness Index
            # Chop > 61.8 = ranging market (mean revert at H5/L5)
            # Chop < 38.2 = trending market (breakout at R1/S1)
            
            if chop > 61.8:
                # Ranging market: mean reversion at H5/L5
                # LONG: Price rejects H5 and moves down with volume
                if price < h5 and price > l5 and vol_ratio_val > 2.0:
                    # Look for rejection of H5 (price below H5 but holding above L5)
                    # Additional confirmation: price closed below midpoint of H5-L5
                    midpoint = (h5 + l5) / 2
                    if price < midpoint:
                        signals[i] = -0.25  # Short at H5 rejection
                        position = -1
                        continue
                
                # SHORT: Price rejects L5 and moves up with volume
                elif price > l5 and price < h5 and vol_ratio_val > 2.0:
                    # Look for rejection of L5 (price above L5 but holding below H5)
                    midpoint = (h5 + l5) / 2
                    if price > midpoint:
                        signals[i] = 0.25   # Long at L5 rejection
                        position = 1
                        continue
                        
            else:
                # Trending market: breakout at R1/S1
                # LONG: Price breaks above R1 with volume
                if price > r1 and vol_ratio_val > 2.0:
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below S1 with volume
                elif price < s1 and vol_ratio_val > 2.0:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Spike"
timeframe = "12h"
leverage = 1.0