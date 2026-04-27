# Based on pattern analysis from successful strategies: 6h timeframe with 1d trend filter
# Focus on Camarilla pivot levels (R3/S3 fade, R4/S4 breakout) combined with volume confirmation
# This approach showed success in DB (1.882 Sharpe for ETHUSDT, 1.844 for ETHUSDT)
# Adjusted for 6h timeframe with proper risk controls

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
    
    # Get daily data for Camarilla pivots and trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Classic formula: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # Using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot range
    pivot_range = prev_high - prev_low
    
    # Camarilla levels
    r4 = prev_close + (pivot_range * 1.1 / 2)
    r3 = prev_close + (pivot_range * 1.1 / 4)
    s3 = prev_close - (pivot_range * 1.1 / 4)
    s4 = prev_close - (pivot_range * 1.1 / 2)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume average for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all indicators ready
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        r4_val = r4_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        s4_val = s4_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.3x daily average
        volume_filter = vol_current > (vol_avg * 1.3)
        
        # Entry conditions
        if position == 0:
            # Long: break above R4 with volume in uptrend, or bounce from S3 with volume in uptrend
            if (close[i] > r4_val or close[i] < s3_val) and volume_filter and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below S4 with volume in downtrend, or bounce from R3 with volume in downtrend
            elif (close[i] < s4_val or close[i] > r3_val) and volume_filter and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend reversal or price reaches opposite level
            if close[i] < ema_trend or close[i] < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend reversal or price reaches opposite level
            if close[i] > ema_trend or close[i] > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_Trend_Volume"
timeframe = "6h"
leverage = 1.0