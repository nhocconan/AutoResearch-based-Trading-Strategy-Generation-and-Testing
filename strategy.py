#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v2
Hypothesis: Camarilla R1/S1 breakout with 1d ATR-based trend filter and volume spike (>3x median) to target 20-40 trades/year. Uses ATR trailing stop (2.0x) for risk management. Designed for low-frequency, high-conviction entries by requiring strong volume confirmation and clear 1d trend alignment using ATR bands, reducing whipsaw in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR(14) for trend filter (price > close + 0.5*ATR = uptrend, < close - 0.5*ATR = downtrend)
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_upper_1d = df_1d['close'].values + 0.5 * atr_14_1d
    atr_lower_1d = df_1d['close'].values - 0.5 * atr_14_1d
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close_1d + (1.0/6) * (prev_high_1d - prev_low_1d)
    camarilla_s1 = prev_close_1d - (1.0/6) * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 4h timeframe
    atr_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_upper_1d)
    atr_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_lower_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike filter: volume > 3x median volume (50-period) for conviction
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    # ATR(14) for volatility-based stops on 4h
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of ATR(14) 1d (50), volume median (50), ATR (14)
    start_idx = max(50, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_upper_1d_aligned[i]) or 
            np.isnan(atr_lower_1d_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        atr_upper_val = atr_upper_1d_aligned[i]
        atr_lower_val = atr_lower_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        # Trend filter: price > ATR upper (uptrend) or < ATR lower (downtrend)
        uptrend = close_val > atr_upper_val
        downtrend = close_val < atr_lower_val
        
        # Volume spike filter: only trade in extreme volume environments
        volume_spike = volume_val > 3.0 * vol_median_val
        
        if position == 0:
            # Long: break above R1 with volume spike, and uptrend
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S1 with volume spike, and downtrend
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0