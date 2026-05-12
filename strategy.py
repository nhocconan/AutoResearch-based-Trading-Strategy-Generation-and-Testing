#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily close price for calculations
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA(34) for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily True Range for volatility calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # Daily Camarilla levels (based on previous day)
    # Using daily OHLC to calculate Camarilla for current day
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    prev_close_1d = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_high_1d = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low_1d = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    # Camarilla levels: R1 = C + ((H-L) * 1.1/12), S1 = C - ((H-L) * 1.1/12)
    camarilla_r1_1d = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 12)
    camarilla_s1_1d = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Volume spike detection: current volume > 2x average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # ensure EMA has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r1_1d_aligned[i]) or 
            np.isnan(camarilla_s1_1d_aligned[i]) or np.isnan(atr10_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + volume spike
            if (close[i] > camarilla_r1_1d_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend + volume spike
            elif (close[i] < camarilla_s1_1d_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below EMA(34) or below S1 (reversal signal)
            if close[i] < ema34_1d_aligned[i] or close[i] < camarilla_s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above EMA(34) or above R1 (reversal signal)
            if close[i] > ema34_1d_aligned[i] or close[i] > camarilla_r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals