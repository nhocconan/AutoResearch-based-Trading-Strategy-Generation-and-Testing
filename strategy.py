#!/usr/bin/env python3
"""
4h_ParabolicSAR_Trend_With_Volume_And_Trend_Filter
Hypothesis: Parabolic SAR signals combined with volume confirmation and 4h EMA trend filter.
Captures trending moves while avoiding whipsaws in sideways markets. Works in both bull and bear regimes
by using the trend filter to align with higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Parabolic SAR calculation
    def parabolic_sar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
        n = len(high)
        sar = np.zeros(n)
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(n)
        ep = np.zeros(n)
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high[0]
        
        for i in range(1, n):
            if trend[i-1] == 1:  # uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                if low[i] <= sar[i]:  # trend reversal
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = low[i]
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                    else:
                        ep[i] = ep[i-1]
                    af[i] = min(af[i-1] + af_increment, af_max)
            else:  # downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                if high[i] >= sar[i]:  # trend reversal
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = high[i]
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                    else:
                        ep[i] = ep[i-1]
                    af[i] = min(af[i-1] + af_increment, af_max)
        return sar, trend
    
    # Calculate Parabolic SAR
    sar, psar_trend = parabolic_sar(high, low)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Trend filter: 4h EMA34
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(sar[i]) or 
            np.isnan(ema_34[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sar_val = sar[i]
        ema_val = ema_34[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price above SAR (uptrend signal) with volume spike and above EMA
            if price > sar_val and vol_spike and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price below SAR (downtrend signal) with volume spike and below EMA
            elif price < sar_val and vol_spike and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below SAR (trend reversal) OR below EMA
            if price < sar_val:
                signals[i] = 0.0
                position = 0
            elif price < ema_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above SAR (trend reversal) OR above EMA
            if price > sar_val:
                signals[i] = 0.0
                position = 0
            elif price > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_ParabolicSAR_Trend_With_Volume_And_Trend_Filter"
timeframe = "4h"
leverage = 1.0