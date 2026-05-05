#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout + 1d EMA34 Trend Filter + Volume Spike
# Long when: price breaks above Camarilla R3 AND 1d EMA34 uptrend AND volume > 2.0x 20-period MA
# Short when: price breaks below Camarilla S3 AND 1d EMA34 downtrend AND volume > 2.0x 20-period MA
# Exit when: price returns to Camarilla Pivot point (PP) OR volume drops below average
# Uses Camarilla levels for structure, EMA34 for trend filter, volume for conviction
# Timeframe: 12h, HTF: 1d for EMA34. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_uptrend = ema_34_1d > np.roll(ema_34_1d, 1)  # EMA rising
    ema34_downtrend = ema_34_1d < np.roll(ema_34_1d, 1)  # EMA falling
    # Handle first value
    ema34_uptrend[0] = False
    ema34_downtrend[0] = False
    
    # Align 1d EMA trend to 12h timeframe
    ema34_uptrend_aligned = align_htf_to_ltf(prices, df_1d, ema34_uptrend.astype(float))
    ema34_downtrend_aligned = align_htf_to_ltf(prices, df_1d, ema34_downtrend.astype(float))
    
    # Calculate Camarilla levels from previous 1d bar (using 1d OHLC)
    # Camarilla: PP = (H+L+C)/3, R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    camarilla_pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    camarilla_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 2
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp.values)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_uptrend_aligned[i]) or np.isnan(ema34_downtrend_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 + uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                ema34_uptrend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S3 + downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema34_downtrend_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to pivot OR volume drops
            if (close[i] <= camarilla_pp_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to pivot OR volume drops
            if (close[i] >= camarilla_pp_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals