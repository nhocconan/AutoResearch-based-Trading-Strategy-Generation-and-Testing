#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume spike
# Works in bull/bear markets due to mean-reversion exit at pivot level
# Target: 20-50 trades/year, avoid overtrading with strict conditions
name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC (for Camarilla calculation)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_range_1d = prev_high_1d - prev_low_1d
    camarilla_r3_1d = camarilla_pivot_1d + camarilla_range_1d * 1.1 / 4
    camarilla_s3_1d = camarilla_pivot_1d - camarilla_range_1d * 1.1 / 4
    
    # Align Camarilla levels to 4h
    camarilla_pivot_4h = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: above 2.0x 12-period average (12*4h = 2 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 12  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above camarilla R3 with 1d uptrend
            if (close[i] > camarilla_r3_4h[i] and 
                close[i] > ema_34_4h[i] and  # 1d uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below camarilla S3 with 1d downtrend
            elif (close[i] < camarilla_s3_4h[i] and 
                  close[i] < ema_34_4h[i] and  # 1d downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals