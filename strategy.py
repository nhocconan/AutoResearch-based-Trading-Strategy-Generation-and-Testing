# [100732] 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT
# Hypothesis: Tight breakout strategy on 12h timeframe using Camarilla R3/S3 levels from 1d data with 1d EMA34 trend filter and volume spike.
# Rationale: Uses tighter R3/S3 levels for fewer, higher-quality trades. 1d trend filter ensures alignment with higher timeframe trend.
# Volume spike confirms breakout strength. Targets 12-37 trades/year to stay within limits and minimize fee drag.
# Works in bull (breakouts with trend) and bear (mean reversion to S3/R3 levels).
# Uses 12h primary timeframe with 1d HTF for both trend and pivot levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day (to avoid look-ahead)
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    daily_r3 = close_1d + daily_range * 1.1 / 4
    daily_s3 = close_1d - daily_range * 1.1 / 4
    
    # Align to 12h timeframe (previous day's levels for current period)
    camarilla_r3 = align_htf_to_ltf(prices, df_1d, daily_r3)
    camarilla_s3 = align_htf_to_ltf(prices, df_1d, daily_s3)
    
    # Volume filter: volume > 2x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R3, above 1d EMA34, volume spike
        if (close[i] > camarilla_r3[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S3, below 1d EMA34, volume spike
        elif (close[i] < camarilla_s3[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite S3/R3 level (mean reversion)
        elif position == 1 and close[i] < camarilla_s3[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > camarilla_r3[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "12h"
leverage = 1.0