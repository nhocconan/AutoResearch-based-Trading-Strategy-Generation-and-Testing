# Hypothesis: 6H_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# - Uses Camarilla pivot levels from 1d for structure
# - Breakout at R3/S3 with 1d EMA trend filter and volume spike confirmation
# - Designed for 6h timeframe to balance trade frequency and signal quality
# - R3/S3 levels offer meaningful breakouts with lower false signals than R4/S4
# - Volume spike (>2x average) filters weak breakouts
# - Should work in both bull (continuation breaks) and bear (mean reversion at extremes)

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
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA 34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Camarilla formula: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    camarilla_width = (high_1d - low_1d) * 1.1 / 2
    r3_1d = close_1d + camarilla_width
    s3_1d = close_1d - camarilla_width
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume filter: volume > 2.0x 24-period average (strong filter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Long conditions: price breaks above R3 + above 1d EMA + volume spike
        long_breakout = (close[i] > r3_1d_aligned[i-1] and price_above_ema and volume_filter[i])
        # Short conditions: price breaks below S3 + below 1d EMA + volume spike
        short_breakout = (close[i] < s3_1d_aligned[i-1] and price_below_ema and volume_filter[i])
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to Camarilla midpoint (pivot)
        elif position == 1 and close[i] < close_1d_aligned[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > close_1d_aligned[i-1]:
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

name = "6H_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0