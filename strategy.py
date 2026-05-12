#!/usr/bin/env python3
name = "1d_Camarilla_R4S4_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    
    # Daily ATR(10) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # Daily previous values for Camarilla calculation
    prev_close_1d = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_high_1d = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low_1d = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    camarilla_r4_1d = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 2)
    camarilla_s4_1d = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to daily timeframe
    camarilla_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Volume spike detection: current volume > 2x average volume (20-day)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure enough data for weekly EMA and other indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(camarilla_r4_1d_aligned[i]) or 
            np.isnan(camarilla_s4_1d_aligned[i]) or np.isnan(atr10_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R4 + weekly uptrend + volume spike
            if (close[i] > camarilla_r4_1d_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + weekly downtrend + volume spike
            elif (close[i] < camarilla_s4_1d_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly EMA or below S4 (reversal)
            if close[i] < ema20_1w_aligned[i] or close[i] < camarilla_s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly EMA or above R4 (reversal)
            if close[i] > ema20_1w_aligned[i] or close[i] > camarilla_r4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals