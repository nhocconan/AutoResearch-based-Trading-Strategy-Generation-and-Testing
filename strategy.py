#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with 1d Volume Spike and 1w Trend Filter
# Long when: price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND 1w close > 1w EMA34
# Short when: price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND 1w close < 1w EMA34
# Exit when price returns to Camarilla Pivot Point (mean reversion)
# Camarilla levels provide intraday support/resistance from prior day
# Volume spike confirms institutional participation
# Weekly EMA34 filter ensures alignment with higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25 to minimize fee churn

name = "6h_Camarilla_R3S3_VolumeSpike_1wTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume average
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1d average volume (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from prior 1d bar
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_range = high_1d - low_1d
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 + volume spike + 1w uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume[i] > 2.0 * vol_ma_aligned[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + volume spike + 1w downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_aligned[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to pivot point (mean reversion)
            if close[i] < camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to pivot point (mean reversion)
            if close[i] > camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals