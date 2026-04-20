#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h chart with 1d ATR-based volatility filter and 1d pivot point breakout.
# Long when price breaks above R1 pivot with volatility contraction (ATR contraction).
# Short when price breaks below S1 pivot with volatility contraction.
# Uses volatility contraction to avoid choppy markets and false breakouts.
# Target: 20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's pivot points (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 14-period ATR on 1d timeframe for volatility filter
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(np.abs(low_1d - np.roll(close_1d, 1)), tr1)
    tr2[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr_val = atr_aligned[i]
        vol_ok = vol_filter[i]
        
        # Volatility contraction filter: current ATR < 0.8 * 20-period ATR average
        atr_ma_20 = pd.Series(atr_aligned).rolling(window=20, min_periods=20).mean().values
        vol_contract = atr_val < 0.8 * atr_ma_20[i] if not np.isnan(atr_ma_20[i]) else False
        
        if position == 0:
            # Long: price breaks above R1, volatility contraction, volume
            if price > r1_val and vol_contract and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, volatility contraction, volume
            elif price < s1_val and vol_contract and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or volatility expansion
            if price < s1_val or not vol_contract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or volatility expansion
            if price > r1_val or not vol_contract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Pivot_R1S1_VolContract_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0