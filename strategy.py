#!/usr/bin/env python3
"""
6h_cci_extreme_1d_vwap_reversion_v1
Hypothesis: Mean-reversion from extreme CCI levels (CCI > 200 or < -200) on 6h, filtered by 1-day VWAP trend.
Long when CCI < -200 (oversold) and price above 1-day VWAP (bullish bias).
Short when CCI > 200 (overbought) and price below 1-day VWAP (bearish bias).
Exit when CCI returns to neutral zone (-100 to 100).
Designed for 12-30 trades/year on 6h with clear logic that works in bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_extreme_1d_vwap_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day VWAP: cumulative(volume * price) / cumulative(volume)
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_num = np.cumsum(typical_price * df_1d['volume'].values)
    vwap_den = np.cumsum(df_1d['volume'].values)
    vwap_1d = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # CCI (20-period) on 6h
    typical = (high + low + close) / 3.0
    sma_tp = pd.Series(typical).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = np.where(mad != 0, (typical - sma_tp) / (0.015 * mad), 0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(cci[i]) or np.isnan(sma_tp[i]) or np.isnan(mad[i])):
            signals[i] = 0.0
            continue
        
        # CCI zones
        cci_overbought = cci[i] > 200
        cci_oversold = cci[i] < -200
        cci_neutral = (cci[i] >= -100) & (cci[i] <= 100)
        
        # VWAP trend filter
        above_vwap = close[i] > vwap_1d_aligned[i]
        below_vwap = close[i] < vwap_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: CCI returns to neutral or price crosses below VWAP
            if cci_neutral or below_vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI returns to neutral or price crosses above VWAP
            if cci_neutral or above_vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: CCI oversold with bullish VWAP bias
            if cci_oversold and above_vwap:
                position = 1
                signals[i] = 0.25
            # Short: CCI overbought with bearish VWAP bias
            elif cci_overbought and below_vwap:
                position = -1
                signals[i] = -0.25
    
    return signals