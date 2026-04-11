#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate CCI(20) on 1d data
    tp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    sma = tp.rolling(window=20, min_periods=20).mean()
    mad = tp.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    cci_values = cci.values
    
    # Align CCI to 4h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_values)
    
    # Volume confirmation: 20-period volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # CCI signals
        cci_buy = cci_aligned[i] > 100
        cci_sell = cci_aligned[i] < -100
        
        # Entry conditions
        # Long: CCI > 100 AND volume confirmation
        if cci_buy and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: CCI < -100 AND volume confirmation
        elif cci_sell and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: CCI crosses back to neutral zone
        elif position == 1 and cci_aligned[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci_aligned[i] > 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals