#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike Confirmation
# Camarilla pivot levels from daily timeframe identify key support/resistance.
# Breakout above R3 or below S3 with volume spike indicates strong momentum.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing strong trends.
# Works in bull markets via long signals in uptrend and bear markets via short signals in downtrend.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF calculations - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1 / 2
    # S3 = Pivot - (H - L) * 1.1 / 2
    # R4 = Pivot + (H - L) * 1.1
    # S4 = Pivot - (H - L) * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 2.0
    r4_1d = pivot_1d + (high_1d - low_1d) * 1.1
    s4_1d = pivot_1d - (high_1d - low_1d) * 1.1
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Break above R3 with volume spike AND 1d uptrend
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_aligned[i]):  # 1d uptrend
                signals[i] = 0.25
                position = 1
            # Short conditions: Break below S3 with volume spike AND 1d downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_aligned[i]):  # 1d downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below R3 OR 1d trend turns down
            if (close[i] < r3_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above S3 OR 1d trend turns up
            if (close[i] > s3_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals