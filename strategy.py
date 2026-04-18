#!/usr/bin/env python3
"""
12h_1w_VWAP_Reversion_with_Volume_Confirmation
Hypothesis: Price reverts to weekly VWAP after significant deviations, with volume confirmation filtering false signals.
Works in bull/bear as mean reversion around institutional value areas. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_numerator = (typical_price * df_1w['volume']).cumsum()
    vwap_denominator = df_1w['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap.values)
    
    # 12-period ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(12, n):
        atr[i] = np.mean(tr[i-11:i+1])
    
    # Volume spike: >2x 24-period average (2 days at 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(volume_spike[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap_1w_aligned[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Deviation from VWAP in ATR units
        if atr_val > 0:
            deviation = (price - vwap_val) / atr_val
        else:
            deviation = 0
        
        if position == 0:
            # Long when price significantly below VWAP with volume spike
            if deviation < -1.5 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when price significantly above VWAP with volume spike
            elif deviation > 1.5 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit when price returns to VWAP or overextends further
            if deviation > -0.5 or deviation < -3.0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit when price returns to VWAP or overextends further
            if deviation < 0.5 or deviation > 3.0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_1w_VWAP_Reversion_with_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0