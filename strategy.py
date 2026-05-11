#!/usr/bin/env python3
name = "4h_Parabolic_SAR_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Parabolic SAR (0.02 step, 0.2 max)
    psar = np.zeros(n)
    psar[0] = low[0]
    psar_bull = True
    af = 0.02
    max_af = 0.2
    ep = high[0] if psar_bull else low[0]
    
    for i in range(1, n):
        if psar_bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if low[i] < psar[i]:
                psar_bull = False
                psar[i] = ep
                af = 0.02
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if high[i] > psar[i]:
                psar_bull = True
                psar[i] = ep
                af = 0.02
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(psar[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above SAR AND above 1d EMA50 AND volume filter
            if close[i] > psar[i] and close[i] > ema_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below SAR AND below 1d EMA50 AND volume filter
            elif close[i] < psar[i] and close[i] < ema_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below SAR OR below 1d EMA50
            if close[i] < psar[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above SAR OR above 1d EMA50
            if close[i] > psar[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals