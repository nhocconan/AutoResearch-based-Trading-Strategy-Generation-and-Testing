#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 with 1d EMA34 uptrend and volume > 1.8x 20-period volume EMA
# Short when price breaks below Camarilla S3 with 1d EMA34 downtrend and volume > 1.8x 20-period volume EMA
# Uses 1d HTF for trend to reduce whipsaw vs shorter HTF, targeting 20-50 trades/year on 4h.
# Volume spike filter (1.8x) is tight to avoid overtrading. Camarilla levels provide clear pivot structure.
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.

name = "4h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using prior 1d bar)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We use the prior completed 1d bar to calculate levels for current 4h bar
    # Since we need prior 1d bar, we shift the 1d data by 1 before calculating
    if len(df_1d) >= 2:
        prev_high_1d = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
        prev_low_1d = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
        prev_close_1d = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    else:
        prev_high_1d = prev_low_1d = prev_close_1d = 0
    
    # Calculate Camarilla levels using prior 1d bar
    H_L = prev_high_1d - prev_low_1d
    camarilla_R3 = prev_close_1d + (H_L * 1.1 / 4)
    camarilla_S3 = prev_close_1d - (H_L * 1.1 / 4)
    
    # Since Camarilla levels are based on prior 1d bar, they are constant within the current 1d bar
    # We need to align these levels to the 4h timeframe
    camarilla_R3_array = np.full(len(prices), camarilla_R3)
    camarilla_S3_array = np.full(len(prices), camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_array)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_array)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.8)  # Volume at least 1.8x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if (close[i] > camarilla_R3_aligned[i] and 
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif (close[i] < camarilla_S3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 1d trend turns down
            if (close[i] < camarilla_S3_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 1d trend turns up
            if (close[i] > camarilla_R3_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals