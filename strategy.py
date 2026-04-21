#!/usr/bin/env python3
"""
4h_Volume_Weighted_CCI_Trend_Filter_V1
Hypothesis: Volume-weighted CCI (VW-CCI) identifies overbought/oversold conditions with institutional participation. 
Trend filter (4h EMA34) ensures trades align with medium-term momentum. 
Volume confirmation filters out low-conviction moves. Works in bull/bear by taking both long and short signals.
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
    
    # VW-CCI: Typical Price * Volume, then standard CCI calculation
    tp = (high + low + close) / 3.0
    vw_tp = tp * volume
    
    # Calculate VW-CCI(20)
    period = 20
    sma_vw_tp = np.full(n, np.nan)
    mad_vw_tp = np.full(n, np.nan)
    
    for i in range(period-1, n):
        sma_vw_tp[i] = np.mean(vw_tp[i-period+1:i+1])
        mad_vw_tp[i] = np.mean(np.abs(vw_tp[i-period+1:i+1] - sma_vw_tp[i]))
    
    vw_cci = np.full(n, np.nan)
    for i in range(period-1, n):
        if mad_vw_tp[i] != 0:
            vw_cci[i] = (vw_tp[i] - sma_vw_tp[i]) / (0.015 * mad_vw_tp[i])
    
    # Trend filter: EMA34 on 4h close
    ema34 = np.full(n, np.nan)
    alpha = 2.0 / (34 + 1)
    for i in range(n):
        if i == 0:
            ema34[i] = close[i]
        else:
            ema34[i] = alpha * close[i] + (1 - alpha) * ema34[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Wait for EMA34 warmup
        if np.isnan(vw_cci[i]) or np.isnan(ema34[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        cci_value = vw_cci[i]
        price = close[i]
        ema = ema34[i]
        
        # Entry conditions
        long_entry = cci_value < -100 and price > ema
        short_entry = cci_value > 100 and price < ema
        
        # Exit conditions
        long_exit = cci_value > 0
        short_exit = cci_value < 0
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Volume_Weighted_CCI_Trend_Filter_V1"
timeframe = "4h"
leverage = 1.0