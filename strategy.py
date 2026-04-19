#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1_S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for volatility filter and stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # Fix first value
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d Close for pivot calculation
    close_1d_vals = df_1d['close'].values
    high_1d_vals = df_1d['high'].values
    low_1d_vals = df_1d['low'].values
    
    # Previous day's OHLC for daily pivot points
    prev_high_1d = np.concatenate([[np.nan], high_1d_vals[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d_vals[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d_vals[:-1]])
    
    # Daily pivot points: P = (H+L+C)/3
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    # Resistance 1: R1 = 2*P - L
    r1_1d = 2 * pivot_1d - prev_low_1d
    # Support 1: S1 = 2*P - H
    s1_1d = 2 * pivot_1d - prev_high_1d
    
    # Align daily pivot levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for volume MA and ATR
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or \
           np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if price > r1_1d_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif price < s1_1d_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: exit if price drops below pivot or ATR-based stop
            if price < pivot_1d_aligned[i] or price < high_since_entry - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Track highest high since entry for trailing stop
                if 'high_since_entry' not in locals():
                    high_since_entry = price
                else:
                    high_since_entry = max(high_since_entry, price)
        
        elif position == -1:
            # Short position: exit if price rises above pivot or ATR-based stop
            if price > pivot_1d_aligned[i] or price > low_since_entry + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Track lowest low since entry for trailing stop
                if 'low_since_entry' not in locals():
                    low_since_entry = price
                else:
                    low_since_entry = min(low_since_entry, price)
    
    return signals