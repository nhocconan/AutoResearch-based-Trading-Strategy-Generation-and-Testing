#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels and volume
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot components from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    R1 = pivot + range_hl * 1.1 / 12
    R2 = pivot + range_hl * 1.1 / 6
    R3 = pivot + range_hl * 1.1 / 4
    R4 = pivot + range_hl * 1.1 / 2
    
    S1 = pivot - range_hl * 1.1 / 12
    S2 = pivot - range_hl * 1.1 / 6
    S3 = pivot - range_hl * 1.1 / 4
    S4 = pivot - range_hl * 1.1 / 2
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Daily average volume (20-period) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 4h ATR for exit (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Pre-compute hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get aligned values
        pivot_val = pivot_aligned[i]
        R1_val = R1_aligned[i]
        R2_val = R2_aligned[i]
        S1_val = S1_aligned[i]
        S2_val = S2_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if np.isnan(pivot_val) or np.isnan(R1_val) or np.isnan(S1_val) or np.isnan(vol_avg) or np.isnan(current_atr):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.3x daily average volume
        vol_spike = current_volume > 1.3 * vol_avg
        
        if position == 0:
            # Long: price breaks above S1 with volume spike (mean reversion bounce)
            if current_close > S1_val and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price breaks below R1 with volume spike (mean reversion fade)
            elif current_close < R1_val and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price reaches pivot or ATR stop loss
            if current_close >= pivot_val:
                signals[i] = 0.0
                position = 0
            elif current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or ATR stop loss
            if current_close <= pivot_val:
                signals[i] = 0.0
                position = 0
            elif current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals