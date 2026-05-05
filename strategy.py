#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when: price breaks above Camarilla R3 AND 1d EMA34 up AND volume > 2.0x 20-period MA
# Short when: price breaks below Camarilla S3 AND 1d EMA34 down AND volume > 2.0x 20-period MA
# Exit when: price retouches Camarilla pivot (PP) OR volume drops below average
# Uses Camarilla levels for structure, 1d EMA for trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for each bar using prior bar's OHLC
    camarilla_pp = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use prior bar's OHLC to avoid look-ahead
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        camarilla_pp[i] = (phigh + plow + pclose) / 3.0
        camarilla_r3[i] = camarilla_pp[i] + (phigh - plow) * 1.1 / 4.0
        camarilla_s3[i] = camarilla_pp[i] - (phigh - plow) * 1.1 / 4.0
    
    # Get 1d data ONCE before loop for EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # EMA trend: 1 = up, -1 = down, 0 = unclear
    ema_trend = np.zeros(len(ema_34), dtype=int)
    for i in range(1, len(ema_34)):
        if not np.isnan(ema_34[i]) and not np.isnan(ema_34[i-1]):
            if ema_34[i] > ema_34[i-1]:
                ema_trend[i] = 1
            elif ema_34[i] < ema_34[i-1]:
                ema_trend[i] = -1
    
    # Align 1d EMA trend to 4h timeframe
    ema_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_trend.astype(float))
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_pp[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_trend_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > R3 AND EMA up AND volume spike
            if (close[i] > camarilla_r3[i] and 
                ema_trend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < S3 AND EMA down AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  ema_trend_aligned[i] == -1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < PP OR volume drops below average
            if (close[i] < camarilla_pp[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > PP OR volume drops below average
            if (close[i] > camarilla_pp[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals