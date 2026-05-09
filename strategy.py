#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    4h TRIX momentum with volume spike and Choppiness regime filter.
    - Long: TRIX crosses above 0 with volume > 2x 20-bar average and CHOP > 61.8 (range)
    - Short: TRIX crosses below 0 with volume > 2x 20-bar average and CHOP > 61.8 (range)
    - Exit: TRIX crosses back through 0 or CHOP < 38.2 (trend regime)
    - Uses volume confirmation to avoid false breakouts
    - Target: 20-40 trades/year on 4h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA of EMA of EMA of ROC)
    roc = np.diff(close, prepend=close[0]) / close
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.values
    
    # Calculate 1d Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    atr_1d[0] = high_1d[0] - low_1d[0]  # first value
    
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum()
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(sum_atr_14 / (max_hh - min_ll)) / np.log10(14)
    chop = chop.values
    chop[np.isnan(chop)] = 50  # neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Long: TRIX crosses above 0 with volume spike in range regime
            if trix[i] > 0 and trix[i-1] <= 0 and vol_ok and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below 0 with volume spike in range regime
            elif trix[i] < 0 and trix[i-1] >= 0 and vol_ok and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below 0 or trend regime (CHOP < 38.2)
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2:  # trend regime - exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above 0 or trend regime (CHOP < 38.2)
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2:  # trend regime - exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals