#!/usr/bin/env python3
"""
1d_7dayVWAP_Deviation_Targeting
Concept: Daily VWAP deviation targeting with 7-day VWAP trend filter and volume confirmation.
- Long: Price < 7-day VWAP AND price > daily VWAP AND daily volume > 1.5x 20-day avg volume
- Short: Price > 7-day VWAP AND price < daily VWAP AND daily volume > 1.5x 20-day avg volume
- Exit: Price crosses daily VWAP
- Position sizing: 0.25
- Works in bull/bear: VWAP mean reversion works in all regimes, volume confirms institutional participation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_7dayVWAP_Deviation_Targeting"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 7-day data ONCE before loop
    df_7d = get_htf_data(prices, '7d')
    if len(df_7d) < 7:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 7-day VWAP ===
    typical_price_7d = (df_7d['high'] + df_7d['low'] + df_7d['close']) / 3.0
    vwap_7d_numerator = (typical_price_7d * df_7d['volume']).cumsum()
    vwap_7d_denominator = df_7d['volume'].cumsum()
    vwap_7d = (vwap_7d_numerator / vwap_7d_denominator).values
    vwap_7d_aligned = align_htf_to_ltf(prices, df_7d, vwap_7d)
    
    # === Daily VWAP ===
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d_numerator = (typical_price_1d * df_1d['volume']).cumsum()
    vwap_1d_denominator = df_1d['volume'].cumsum()
    vwap_1d = (vwap_1d_numerator / vwap_1d_denominator).values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === Daily Volume Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # === Daily Price ===
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        vwap_7d_val = vwap_7d_aligned[i]
        vwap_1d_val = vwap_1d_aligned[i]
        vol_ma_val = vol_ma_20_1d_aligned[i]
        current_vol = volume_1d_aligned[i]
        price = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(vwap_7d_val) or np.isnan(vwap_1d_val) or 
            np.isnan(vol_ma_val) or np.isnan(current_vol)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-day average
        vol_condition = current_vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price below 7-day VWAP but above daily VWAP with volume spike
            if price < vwap_7d_val and price > vwap_1d_val and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price above 7-day VWAP but below daily VWAP with volume spike
            elif price > vwap_7d_val and price < vwap_1d_val and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily VWAP
            if price < vwap_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily VWAP
            if price > vwap_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals