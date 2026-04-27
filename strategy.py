# 6H_1d_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 6h timeframe, trade Camarilla pivot breakouts (R3/S3) in the direction of daily trend with volume confirmation.
# Works in bull/bear because trend filter adapts and volume ensures momentum. Targets 12-37 trades/year via strict R3/S3 breakout + trend + volume confluence.
# Uses 1d Camarilla levels for structure, 1d EMA34 for trend, and volume spike for confirmation.

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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range and Camarilla levels
    daily_range = high_1d - low_1d
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    r3 = close_1d + daily_range * 1.1 / 2
    s3 = close_1d - daily_range * 1.1 / 2
    # R4/S4 for additional context (not used for entry but for trend alignment)
    r4 = close_1d + daily_range * 1.1
    s4 = close_1d - daily_range * 1.1
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily average volume for volume spike filter
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need 34 periods for EMA + 20 for volume average
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0x daily average (spike)
        volume_filter = vol_current > (vol_avg * 2.0)
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume spike
            if close[i] > r3_val and close[i] > ema_trend and volume_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 + downtrend + volume spike
            elif close[i] < s3_val and close[i] < ema_trend and volume_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S3 or trend reversal
            if close[i] < s3_val or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R3 or trend reversal
            if close[i] > r3_val or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6H_1d_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0