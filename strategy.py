#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with volume spike confirmation and 1w EMA50 trend filter
# Uses 1d Camarilla levels for institutional support/resistance
# Uses 1w EMA50 to filter for major trend direction (avoid counter-trend trades)
# Uses 1d volume > 1.8x 20-period EMA for confirmation
# Designed for 1d timeframe targeting 15-25 trades/year with discrete sizing (0.30)
# Volume spike + 1w trend filter reduces false breakouts while maintaining trend alignment
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend)

name = "1d_Camarilla_R3S3_Breakout_VolumeSpike_1wEMA50"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume EMA(20) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_series = pd.Series(vol_1d)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate camarilla levels: R3, S3 from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 2
    s3 = close_1d - 1.1 * camarilla_range / 2
    
    # Align camarilla levels to 1d timeframe (same timeframe, so direct assignment)
    r3_aligned = r3  # Already 1d data
    s3_aligned = s3  # Already 1d data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA50
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.8 x 20-period EMA
        volume_confirmed = volume[i] > (1.8 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: close breaks above R3 + volume confirmation + price > 1w EMA50 (uptrend)
            if (close[i] > r3_aligned[i] and volume_confirmed and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: close breaks below S3 + volume confirmation + price < 1w EMA50 (downtrend)
            elif (close[i] < s3_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 (mean reversion) OR price < 1w EMA50 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above R3 (mean reversion) OR price > 1w EMA50 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals