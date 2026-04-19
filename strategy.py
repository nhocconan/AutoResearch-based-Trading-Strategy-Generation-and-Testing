#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1w Pivot R2/S2 breakout, volume confirmation, and ATR-based volatility filter.
# Uses weekly pivot levels as dynamic support/resistance, enters on breakouts with volume and volatility confirmation.
# Designed to work in both bull and bear markets by filtering for genuine momentum bursts.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
name = "6h_1w_Pivot_R2S2_Breakout_VolumeATRFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Pivot calculation (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Pivot, R2, S2 on 1w timeframe
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r2_1w = pivot_1w + (high_1w - low_1w)  # R2 = P + (H - L)
    s2_1w = pivot_1w - (high_1w - low_1w)  # S2 = P - (H - L)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Volume spike (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for ATR and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        # Volatility filter: require ATR > 0.5 * 50-period ATR average
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > (0.5 * atr_ma[i]) if not np.isnan(atr_ma[i]) else True
        
        if position == 0:
            # Long when price breaks above R2 with volume and volatility
            if close[i] > r2_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S2 with volume and volatility
            elif close[i] < s2_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below S2 (reversal) or volatility drops
            if close[i] < s2_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R2 (reversal) or volatility drops
            if close[i] > r2_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals