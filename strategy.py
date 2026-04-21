#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:14])
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, close_1d + 1.5 * atr)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, close_1d - 1.5 * atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (4h close and 1d volume aligned)
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        if position == 0:
            # Enter long: price breaks above 1d ATR-based upper band + volume surge
            if (price_close > upper_band_aligned[i] and
                vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 1d ATR-based lower band + volume surge
            elif (price_close < lower_band_aligned[i] and
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite band or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price < lower band or volume < average
                if (price_close < lower_band_aligned[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price > upper band or volume < average
                if (price_close > upper_band_aligned[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_ATRBreakout_VolumeSurge"
timeframe = "4h"
leverage = 1.0