#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (long-term trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA50 for 12h trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_R1 = np.zeros(len(df_1d))
    camarilla_S1 = np.zeros(len(df_1d))
    camarilla_R3 = np.zeros(len(df_1d))
    camarilla_S3 = np.zeros(len(df_1d))
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    camarilla_R1 = pivot + range_hl * 1.1 / 12
    camarilla_S1 = pivot - range_hl * 1.1 / 12
    camarilla_R3 = pivot + range_hl * 1.1 / 4
    camarilla_S3 = pivot - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume spike detection (20-period average)
    volume_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    volume_ma[:10] = volume_ma[10]
    volume_ma[-10:] = volume_ma[-11]
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Camarilla R1 AND 12h EMA50 uptrend AND volume spike
            long_cond = (close[i] > camarilla_R1_aligned[i]) and (close[i] > ema50_12h_aligned[i]) and volume_spike[i]
            
            # Short entry: price breaks below Camarilla S1 AND 12h EMA50 downtrend AND volume spike
            short_cond = (close[i] < camarilla_S1_aligned[i]) and (close[i] < ema50_12h_aligned[i]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 (reversal) OR 12h EMA50 downtrend
            if (close[i] < camarilla_S1_aligned[i]) or (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 (reversal) OR 12h EMA50 uptrend
            if (close[i] > camarilla_R1_aligned[i]) or (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 levels act as intraday support/resistance. Breaking these levels with volume
# confirmation indicates a strong move. The 12h EMA50 filter ensures we only trade in the direction of the
# longer-term trend, avoiding counter-trend trades during choppy periods. This strategy works in bull
# markets (trend continuation) and bear markets (trend reversals when price breaks S1/R1 with volume).
# Target: 20-50 trades per year to stay within the optimal range for 4h strategies.