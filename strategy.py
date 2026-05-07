# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Price breaking above Camarilla R3 or below S3 with 1-day trend filter and volume confirmation captures strong moves in both bull and bear markets. Low-frequency signals via 12h timeframe with confluence of Camarilla pivot levels, trend, and volume.
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate 12h Camarilla pivot levels: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    # Using daily high/low/close for Camarilla calculation
    daily_range = high - low
    camarilla_r3 = close + daily_range * 1.1 / 2.0
    camarilla_s3 = close - daily_range * 1.1 / 2.0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + 1d uptrend + volume
            if close[i] > camarilla_r3[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + 1d downtrend + volume
            elif close[i] < camarilla_s3[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite Camarilla level
            if position == 1:
                if close[i] < camarilla_s3[i]:  # Exit long if price breaks below S3
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > camarilla_r3[i]:  # Exit short if price breaks above R3
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals