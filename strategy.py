# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# 12h timeframe with Camarilla pivot breakout from daily timeframe, filtered by daily trend and volume spike
# Camarilla provides precise intraday levels, trend filter avoids counter-trend trades, volume confirms momentum
# Targets 15-30 trades per year (60-120 total over 4 years) for low fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_R3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_S3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Daily trend filter: EMA34 slope
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_slope = ema34_1d[1:] - ema34_1d[:-1]  # positive = uptrend
    ema34_slope = np.concatenate([[0], ema34_slope])  # align length
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_slope)
    
    # Volume confirmation: current volume > 2.0x 24-period average (2 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(ema34_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        r3_level = camarilla_R3_aligned[i]
        s3_level = camarilla_S3_aligned[i]
        trend_slope = ema34_slope_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike and daily uptrend
            if close_val > r3_level and vol_spike_val and trend_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike and daily downtrend
            elif close_val < s3_level and vol_spike_val and trend_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or daily trend turns down
            if close_val < s3_level or trend_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or daily trend turns up
            if close_val > r3_level or trend_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals