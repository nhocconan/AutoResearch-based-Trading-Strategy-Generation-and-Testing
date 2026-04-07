#!/usr/bin/env python3
"""
4h_cci_breakout_12h_volatility_v1
Hypothesis: On 4h timeframe, use 12h CCI (Commodity Channel Index) to detect extreme market conditions (CCI > 100 or < -100) combined with 12h volatility regime (ATR-based) to filter entries. Enter long when CCI crosses above 100 with volatility > 20-period average; enter short when CCI crosses below -100 with volatility > 20-period average. Exit when CCI crosses back through zero. This strategy captures momentum bursts during high volatility periods, which often precede sustained moves in both bull and bear markets. The 12h CCI filters out noise, and volatility regime ensures we only trade when there's sufficient market participation. Targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_breakout_12h_volatility_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h CCI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Typical price for CCI
    tp_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    tp_12h_values = tp_12h.values
    
    # CCI calculation: (TP - SMA(TP,20)) / (0.015 * MeanDeviation(TP,20))
    sma_tp = pd.Series(tp_12h_values).rolling(window=20, min_periods=20).mean().values
    mad_tp = pd.Series(tp_12h_values).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    mad_tp = np.where(mad_tp == 0, 0.001, mad_tp)
    cci_12h = (tp_12h_values - sma_tp) / (0.015 * mad_tp)
    
    # Align CCI to 4h timeframe
    cci_12h_aligned = align_htf_to_ltf(prices, df_12h, cci_12h)
    
    # Calculate 12h ATR for volatility regime
    tr1_12h = df_12h['high'] - df_12h['low']
    tr2_12h = np.abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3_12h = np.abs(df_12h['low'] - df_12h['close'].shift(1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volatility filter: ATR > 20-period average
    atr_ma = pd.Series(atr_12h_aligned).rolling(window=20, min_periods=20).mean().values
    vol_regime = atr_12h_aligned > atr_ma
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(cci_12h_aligned[i]) or np.isnan(vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # CCI crossing above 100 (long entry)
        long_entry = (cci_12h_aligned[i] > 100 and cci_12h_aligned[i-1] <= 100 and vol_regime[i])
        # CCI crossing below -100 (short entry)
        short_entry = (cci_12h_aligned[i] < -100 and cci_12h_aligned[i-1] >= -100 and vol_regime[i])
        # CCI crossing zero (exit)
        exit_long = (cci_12h_aligned[i] < 0 and cci_12h_aligned[i-1] >= 0)
        exit_short = (cci_12h_aligned[i] > 0 and cci_12h_aligned[i-1] <= 0)
        
        # Track position state
        if i == 50:
            position = 0
        else:
            position = 1 if signals[i-1] > 0 else (-1 if signals[i-1] < 0 else 0)
        
        if position == 1:  # Long position
            if exit_long:
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            if exit_short:
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if long_entry:
                signals[i] = 0.25
            elif short_entry:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals