#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot + Daily Trend + Volume Spike
# Hypothesis: Camarilla pivot levels from daily chart provide strong support/resistance.
# Breakouts from these levels with daily trend alignment and volume confirmation
# capture institutional flow. Works in both bull/bear by using pivot-based structure.
# Target: 25-35 trades/year (100-140 total).

name = "4h_camarilla_pivot_daily_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use R3, R2, S3, S2 for breakouts
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_r2 = close_1d + range_1d * 1.1 / 6.0
    camarilla_s2 = close_1d - range_1d * 1.1 / 6.0
    camarilla_s3 = close_1d - range_1d * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S2 or trend turns bearish
            if close[i] < s2_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above R2 or trend turns bullish
            if close[i] > r2_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above R3 in uptrend
                if close[i] > r3_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below S3 in downtrend
                elif close[i] < s3_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals