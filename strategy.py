# 6h_Pivot_R1_S1_Breakout_Volume_ATRFilter
# Uses 1-day Camarilla pivot levels (R1/S1) with volume and ATR filter for 6h timeframe
# Long: price breaks above R1 with volume spike and ATR-based momentum
# Short: price breaks below S1 with volume spike and ATR-based momentum
# Exit: price returns to pivot point (PP) or ATR-based stop
# Designed for 50-150 total trades over 4 years (12-37/year) with controlled risk

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align pivot levels to 6h timeframe (using previous day's values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 1.5 x 20-period average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    # ATR for momentum filter and stop sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Price momentum: close > open (bullish) or close < open (bearish)
    bullish_momentum = close > prices['open'].values
    bearish_momentum = close < prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical indicators
        if np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(volume_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        pp_level = pp_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and bullish momentum
            if price > r1_level and volume_spike[i] and bullish_momentum[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and bearish momentum
            elif price < s1_level and volume_spike[i] and bearish_momentum[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point or ATR-based stop
            if price <= pp_level or price <= prices['open'].values[i] - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point or ATR-based stop
            if price >= pp_level or price >= prices['open'].values[i] + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0