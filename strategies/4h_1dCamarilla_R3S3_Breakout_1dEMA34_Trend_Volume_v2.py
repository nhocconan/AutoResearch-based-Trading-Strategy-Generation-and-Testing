# 4h_1dCamarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v2
# Uses daily Camarilla pivot levels (R3/S3) as breakout levels with daily trend filter (EMA34)
# and daily volume confirmation. Designed for 4h timeframe to capture major pivot breaks
# with trend alignment, working in both bull and bear markets by following the daily trend.
# Target: 75-200 total trades over 4 years (19-50/year) with 0.30 position sizing.

name = "4h_1dCamarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pp + range_1d * 1.1 / 2
    s3 = pp - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume filter (20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with uptrend and volume
            if close[i] > r3_4h[i] and close[i] > ema_34_4h[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: break below S3 with downtrend and volume
            elif close[i] < s3_4h[i] and close[i] < ema_34_4h[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price returns to EMA34 or breaks below S3
            if close[i] < ema_34_4h[i] or close[i] < s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price returns to EMA34 or breaks above R3
            if close[i] > ema_34_4h[i] or close[i] > r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals