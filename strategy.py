#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout + 1d EMA34 Trend + Volume Spike
# Long when: Close breaks above R3 (1d) AND 12h EMA > 1d EMA34 (uptrend) AND volume > 2x 20-period MA
# Short when: Close breaks below S3 (1d) AND 12h EMA < 1d EMA34 (downtrend) AND volume > 2x 20-period MA
# Exit when: Close retraces to Camarilla pivot point (PP) OR EMA trend flips
# Uses proven Camarilla structure with 12h/1d alignment for BTC/ETH resilience in bull/bear
# Timeframe: 12h, HTF: 1d for EMA34 and Camarilla levels. Target: 50-150 total trades over 4 years.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA(34) on 12h for trend
    if len(close) >= 34:
        ema_34_12h = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    else:
        ema_34_12h = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need sufficient data for EMA34 and Camarilla
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    if len(close_1d) >= 34:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    else:
        ema_34_1d = np.full(len(df_1d), np.nan)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_pp = np.zeros(len(df_1d))
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_pp[i] = np.nan
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
        else:
            phigh = high_1d[i-1]
            plow = low_1d[i-1]
            pclose = close_1d[i-1]
            camarilla_pp[i] = (phigh + plow + pclose) / 3.0
            camarilla_r3[i] = pclose + (phigh - plow) * 1.1 / 2.0
            camarilla_s3[i] = pclose - (phigh - plow) * 1.1 / 2.0
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Align 1d indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Trend filter: 12h EMA > 1d EMA34 = uptrend, < = downtrend
    ema_trend_up = ema_34_12h > ema_34_1d_aligned
    ema_trend_down = ema_34_12h < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_trend_up[i]) or np.isnan(ema_trend_down[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pp_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: close breaks above R3 + uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_trend_up[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: close breaks below S3 + downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_trend_down[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close retraces to PP OR trend flips down
            if (close[i] <= camarilla_pp_aligned[i] or not ema_trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close retraces to PP OR trend flips up
            if (close[i] >= camarilla_pp_aligned[i] or not ema_trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals