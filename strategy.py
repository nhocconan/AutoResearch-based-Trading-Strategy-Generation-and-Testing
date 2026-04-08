#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # TRIX: Triple Exponential Moving Average (15-period)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # Percentage change
    trix_values = trix.values
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_values)
    
    # TRIX Signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Volume filter: current volume > 1.3x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 1.3)
    
    # Volatility filter: ATR(14) > 20-period ATR mean (avoid choppy markets)
    tr1 = pd.Series(high).subtract(pd.Series(low)).abs()
    tr2 = pd.Series(high).subtract(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).subtract(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > atr_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_1d_aligned[i]) or 
            np.isnan(trix_signal_aligned[i]) or
            np.isnan(vol_spike[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below signal line
            if trix_1d_aligned[i] < trix_signal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above signal line
            if trix_1d_aligned[i] > trix_signal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: TRIX crosses above signal line + volume spike + vol filter
            if (trix_1d_aligned[i] > trix_signal_aligned[i] and 
                trix_1d_aligned[i-1] <= trix_signal_aligned[i-1] and
                vol_spike[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: TRIX crosses below signal line + volume spike + vol filter
            elif (trix_1d_aligned[i] < trix_signal_aligned[i] and 
                  trix_1d_aligned[i-1] >= trix_signal_aligned[i-1] and
                  vol_spike[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals