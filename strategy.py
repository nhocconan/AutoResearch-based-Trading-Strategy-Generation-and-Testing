#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout with 1w EMA50 trend filter and volume confirmation (2.0x)
# Long when price breaks above R3 AND price > 1w EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below S3 AND price < 1w EMA50 AND volume > 2.0x 20-period average
# Exit when price reverts to Camarilla pivot point (PP)
# Uses 12h timeframe with 1w HTF for robust trend filtering effective in both bull and bear markets
# Weekly EMA50 provides strong trend filter that avoids whipsaws in ranging markets
# Volume confirmation at 2.0x reduces false breakouts while maintaining sufficient trade frequency
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_2.0x"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels from previous 1w bar
    camarilla_high = np.roll(high_1w, 1)  # previous 1w bar high
    camarilla_low = np.roll(low_1w, 1)    # previous 1w bar low
    camarilla_close = np.roll(close_1w, 1) # previous 1w bar close
    
    # Calculate Camarilla levels
    camarilla_range = camarilla_high - camarilla_low
    r3 = camarilla_close + 1.1 * camarilla_range
    s3 = camarilla_close - 1.1 * camarilla_range
    pp = (camarilla_high + camarilla_low + camarilla_close) / 3.0
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation on 12h (threshold: 2.0x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND price > EMA50 AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND price < EMA50 AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to PP
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to PP
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals