#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
Breakout above R1 or below S1 with volume confirmation and ATR-based trend filter.
Works in bull/bear: trend filter prevents counter-trend trades, volatility filter avoids chop.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily: ATR for trend filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # === Daily: Camarilla pivot levels (R1, S1) ===
    # Formula: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    hl_range = high_1d - low_1d
    r1 = close_1d + 1.1 * hl_range / 12
    s1 = close_1d - 1.1 * hl_range / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr14_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(atr_val) or np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with volume and above ATR-based trend
            if (close_val > r1_val and
                vol_ratio_val > 1.5 and
                close_val > close[i-20] + 0.5 * atr_val):  # upward momentum
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and below ATR-based trend
            elif (close_val < s1_val and
                  vol_ratio_val > 1.5 and
                  close_val < close[i-20] - 0.5 * atr_val):  # downward momentum
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below S1 or loss of momentum
            if (close_val < s1_val or
                close_val < close[i-5]):  # recent momentum loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above R1 or loss of momentum
            if (close_val > r1_val or
                close_val > close[i-5]):  # recent momentum loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals