# 12h_Camarilla_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance zones.
# Price breaking above/below key Camarilla levels (S3/R3) with volume confirmation and
# aligned with daily trend (EMA34) yields high-probability trades. Works in both bull and bear
# markets as it trades breakouts from institutional levels with volume confirmation.
# Target: 12h timeframe, 20-40 trades/year, low frequency to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from daily OHLC
    # Camarilla: H-L = range, then levels = C ± (range * multiplier)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    daily_range = high_1d - low_1d
    camarilla_r3 = close_1d_arr + (daily_range * 1.1/2)  # R3 = C + (range * 1.1/2)
    camarilla_s3 = close_1d_arr - (daily_range * 1.1/2)  # S3 = C - (range * 1.1/2)
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: volume > 2.0x 20-period average (strong confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + volume + daily uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume + daily downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses back below R3 or daily trend turns down
            if (close[i] < camarilla_r3_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above S3 or daily trend turns up
            if (close[i] > camarilla_s3_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0