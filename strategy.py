# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Breakout of Camarilla R3/S3 levels with 1d trend filter and volume spike.
# Uses daily trend (EMA34) and volume > 2x 20-period average for confirmation.
# Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in bull (breakouts) and bear (breakdowns) via symmetry.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Previous day OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for current 4h bar using previous day's OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = prev_day_close + (prev_day_high - prev_day_low) * 1.1 / 2.0
    camarilla_s3 = prev_day_close - (prev_day_high - prev_day_low) * 1.1 / 2.0
    
    # Align daily levels to 4h timeframe (waits for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d trend filter: EMA34 slope
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_slope = ema34_1d[1:] - ema34_1d[:-1]
    ema34_1d_slope = np.concatenate([[0], ema34_1d_slope])
    ema34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_slope)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema34_1d_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema_slope = ema34_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: break above R3 with 1d uptrend and volume spike
            if close[i] > r3 and ema_slope > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with 1d downtrend and volume spike
            elif close[i] < s3 and ema_slope < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or trend turns down
            if close[i] < r3 or ema_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or trend turns up
            if close[i] > s3 or ema_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals