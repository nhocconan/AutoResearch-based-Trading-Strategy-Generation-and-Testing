#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Long when: price breaks above R3 AND close > 4h EMA50 AND volume > 2x 20-period MA
# Short when: price breaks below S3 AND close < 4h EMA50 AND volume > 2x 20-period MA
# Exit when: price returns to Pivot point (PP) OR volume drops below average
# Uses Camarilla levels for precise entry/exit, 4h EMA for trend alignment, volume for conviction
# Timeframe: 1h, HTF: 4h. Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels on 1h using previous bar's OHLC
    camarilla_high = np.roll(high, 1)
    camarilla_low = np.roll(low, 1)
    camarilla_close = np.roll(close, 1)
    camarilla_high[0] = camarilla_low[0] = camarilla_close[0] = np.nan
    
    camarilla_range = camarilla_high - camarilla_low
    camarilla_pp = (camarilla_high + camarilla_low + camarilla_close) / 3
    camarilla_r3 = camarilla_pp + (camarilla_range * 1.1 / 4)
    camarilla_s3 = camarilla_pp - (camarilla_range * 1.1 / 4)
    
    # Get 4h data ONCE before loop for EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h close
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation on 1h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pp[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 + above 4h EMA50 + volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: break below S3 + below 4h EMA50 + volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: return to Pivot point OR volume drops below average
            if (close[i] <= camarilla_pp[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: return to Pivot point OR volume drops below average
            if (close[i] >= camarilla_pp[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals