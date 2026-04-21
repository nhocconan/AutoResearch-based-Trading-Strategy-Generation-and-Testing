#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Confluence_v1
Hypothesis: On 6h timeframe, price breaking above Donchian(20) high or below Donchian(20) low captures momentum breakouts. Combined with weekly Camarilla pivot direction (from prior week) for regime filter and volume spike confirmation. Weekly pivot provides multi-day structure that works in both bull (continuation at R4/S4) and bear (fade at R3/S3) regimes. Designed for low trade frequency (~15-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Donchian, 1w for weekly Camarilla)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 5:
        return np.zeros(n)
    
    # === 1d Donchian(20) for breakout ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high/low: 20-period rolling max/min
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1w Camarilla levels from prior weekly session ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R3, S3, R4, S4
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    camarilla_r4 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_s4 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3, additional_delay_bars=0)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3, additional_delay_bars=0)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4, additional_delay_bars=0)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4, additional_delay_bars=0)
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly R4 (bullish continuation) + volume spike
            if price_close > donch_high and price_close > r4 and vol_spike > 2.0:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below Donchian low + below weekly S4 (bearish continuation) + volume spike
            elif price_close < donch_low and price_close < s4 and vol_spike > 2.0:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Time-based exit: exit after 3 bars (18 hours) to avoid overtrading
            # Simple approach: flip to flat after holding period
            # For more sophisticated exit, could use opposite Donchian break
            signals[i] = 0.25 if position == 1 else -0.25
            
            # Exit condition: holding for 3 bars
            # Track bars held since entry (simplified: use position duration)
            # For simplicity, exit on opposite Donchian break or after 3 bars
            # We'll implement a simple time-based exit: exit after 3 bars
            # Since we don't track entry bar, use a counter approach
            
            # Instead, use opposing Donchian break for exit
            if position == 1:
                if price_close < donch_low:  # Opposite break
                    signals[i] = 0.0
                    position = 0
            else:  # position == -1
                if price_close > donch_high:  # Opposite break
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Confluence_v1"
timeframe = "6h"
leverage = 1.0