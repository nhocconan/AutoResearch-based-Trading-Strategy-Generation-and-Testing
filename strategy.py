#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_RMAD_Trend_4hVolFilter_1dTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RMAD trend and volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate RMAD (Robust Moving Average Deviation) on 4h close
    close_4h = df_4h['close'].values
    # RMAD = median of absolute deviations from median over window
    def rolling_median(arr, window):
        from scipy.ndimage import median_filter
        return median_filter(arr, size=window, mode='constant', cval=np.nan)
    
    # Use 20-period median for trend
    median_4h = rolling_median(close_4h, 20)
    # Calculate absolute deviation from median
    abs_dev = np.abs(close_4h - median_4h)
    # RMAD = median of absolute deviations
    rm_abs_dev = rolling_median(abs_dev, 20)
    # RMAD indicator: price position relative to median
    rm_ad = (close_4h - median_4h) / (rm_abs_dev + 1e-10)
    # Smooth the RMAD
    rm_ad_smooth = pd.Series(rm_ad).ewm(span=10, adjust=False, min_periods=10).mean().values
    rm_ad_aligned = align_htf_to_ltf(prices, df_4h, rm_ad_smooth)
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = vol_4h / vol_ma_4h
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # 1d trend filter: EMA(50) slope
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope_1d = np.diff(ema_50_1d, prepend=ema_50_1d[0])
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rm_ad_aligned[i]) or np.isnan(vol_ratio_4h_aligned[i]) or 
            np.isnan(ema_slope_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RMAD > 0 (above median) + high volume + rising 1d EMA
            if (rm_ad_aligned[i] > 0.1 and vol_ratio_4h_aligned[i] > 1.5 and 
                ema_slope_1d_aligned[i] > 0):
                signals[i] = 0.20
                position = 1
            # Short: RMAD < 0 (below median) + high volume + falling 1d EMA
            elif (rm_ad_aligned[i] < -0.1 and vol_ratio_4h_aligned[i] > 1.5 and 
                  ema_slope_1d_aligned[i] < 0):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RMAD crosses below 0 or volume drops
            if (rm_ad_aligned[i] < -0.05 or vol_ratio_4h_aligned[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RMAD crosses above 0 or volume drops
            if (rm_ad_aligned[i] > 0.05 or vol_ratio_4h_aligned[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals