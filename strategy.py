#!/usr/bin/env python3
# 4h_12h_volume_crossover_v1
# Hypothesis: 4-hour price momentum confirmed by 12-hour volume surge and moving average crossovers.
# Long when 4h EMA(21) crosses above EMA(55) with 12h volume > 1.5x 20-period average.
# Short when 4h EMA(21) crosses below EMA(55) with 12h volume > 1.5x 20-period average.
# Uses volume confirmation to avoid false breakouts and reduce whipsaw.
# Designed for 20-40 trades/year on 4h to minimize fee drag while capturing trending moves.
# Works in bull markets via momentum captures and bear markets via short signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_volume_crossover_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA(21) and EMA(55) for crossover signals
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Get 12h volume data for confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Pre-compute 12h volume moving average and align
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    vol_current_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 55  # Ensure EMA(55) is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema21[i]) or np.isnan(ema55[i]) or np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_current_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current 12h volume > 1.5x 20-period average
        vol_surge = vol_current_aligned[i] > 1.5 * vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: EMA(21) crosses below EMA(55)
            if ema21[i] < ema55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA(21) crosses above EMA(55)
            if ema21[i] > ema55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA(21) crosses above EMA(55) with volume surge
            if ema21[i] > ema55[i] and ema21[i-1] <= ema55[i-1] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: EMA(21) crosses below EMA(55) with volume surge
            elif ema21[i] < ema55[i] and ema21[i-1] >= ema55[i-1] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals