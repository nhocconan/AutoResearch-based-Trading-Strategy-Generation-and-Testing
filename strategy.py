#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot level touch with 1d trend filter and volume confirmation.
# Long when price touches S1 support with 1d EMA34 uptrend and volume spike.
# Short when price touches R1 resistance with 1d EMA34 downtrend and volume spike.
# Exit when price reaches opposite pivot level (S1/R1) or closes back inside H-L range.
# Uses proven Camarilla pivot structure with strict entry conditions to limit trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # S1 = C - (H-L)*1.12, R1 = C + (H-L)*1.12
    camarilla_s1 = np.full(len(close_1d), np.nan)
    camarilla_r1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's H, L, C to calculate today's levels
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        rang = phigh - plow
        if rang > 0:
            camarilla_s1[i] = pclose - (rang * 1.12)
            camarilla_r1[i] = pclose + (rang * 1.12)
    
    # Align 1d data to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    
    # Volume spike filter (volume > 2x 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA34 and volume MA20
    start_idx = max(ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume spike (2x average)
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price touches S1 support with 1d EMA34 uptrend and volume spike
            if (price <= camarilla_s1_aligned[i] * 1.001 and  # Allow small tolerance
                price >= camarilla_s1_aligned[i] * 0.999 and
                ema_1d_aligned[i] > ema_1d_aligned[i-1] and  # Rising EMA = uptrend
                vol_filter):
                signals[i] = size
                position = 1
            # Short: price touches R1 resistance with 1d EMA34 downtrend and volume spike
            elif (price >= camarilla_r1_aligned[i] * 0.999 and  # Allow small tolerance
                  price <= camarilla_r1_aligned[i] * 1.001 and
                  ema_1d_aligned[i] < ema_1d_aligned[i-1] and  # Falling EMA = downtrend
                  vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R1 or closes back inside H-L range of pivot
            if (price >= camarilla_r1_aligned[i] * 0.999 or  # Hit R1 target
                (low[i] >= camarilla_s1_aligned[i] and high[i] <= camarilla_r1_aligned[i])):  # Back inside range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches S1 or closes back inside H-L range of pivot
            if (price <= camarilla_s1_aligned[i] * 1.001 or  # Hit S1 target
                (low[i] >= camarilla_s1_aligned[i] and high[i] <= camarilla_r1_aligned[i])):  # Back inside range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_S1R1_Touch_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0