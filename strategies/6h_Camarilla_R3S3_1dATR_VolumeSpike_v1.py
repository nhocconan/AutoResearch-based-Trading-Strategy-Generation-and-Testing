#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d ATR volatility filter and volume spike
# Uses Camarilla pivot levels from 1d for structure, 1d ATR(14) to filter low-volatility chop
# Volume spike ensures participation and reduces false breakouts
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by only taking breakouts with volume confirmation
# ATR filter avoids whipsaws in ranging markets

name = "6h_Camarilla_R3S3_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = pivot + range_1d * 1.1 / 2.0
    s3 = pivot - range_1d * 1.1 / 2.0
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Camarilla levels and ATR to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 6h volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for ATR and volume MA)
    start_idx = 34  # max(14 for ATR, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: only trade when volatility is above average (avoid chop)
        # Use 1.0 as threshold - trade when ATR > its own value (always true when not NaN)
        # Instead, use ATR ratio to avoid low volatility periods
        atr_ma = pd.Series(atr_14_aligned).rolling(window=10, min_periods=10).mean().shift(1).values
        if np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            volatility_filter = True  # allow trade if MA not ready
        else:
            volatility_filter = atr_14_aligned[i] > (atr_ma[i] * 0.8)  # trade when ATR > 80% of MA
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND volume confirm AND volatility filter
            if (close[i] > r3_aligned[i] and 
                volume_confirm[i] and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume confirm AND volatility filter
            elif (close[i] < s3_aligned[i] and 
                  volume_confirm[i] and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 OR volatility drops significantly
            if (close[i] < s3_aligned[i] or 
                (not volatility_filter and atr_14_aligned[i] < (atr_ma[i] * 0.6))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR volatility drops significantly
            if (close[i] > r3_aligned[i] or 
                (not volatility_filter and atr_14_aligned[i] < (atr_ma[i] * 0.6))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals