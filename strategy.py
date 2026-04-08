# 12h_camarilla_pivot_daily_trend_volume_v1
# Hypothesis: Uses daily Camarilla pivot levels with volume confirmation and 1d trend filter.
# Goes long when price touches S3 level in uptrend (price > 1d EMA50) with volume spike.
# Goes short when price touches R3 level in downtrend (price < 1d EMA50) with volume spike.
# Uses 12h timeframe for lower trade frequency and better risk-adjusted returns in both bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 2:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    # We need previous day's data, so shift by 1
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first value to NaN since no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.2500)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.2500)
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Need enough data for volume average
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price moves above S2 or trend changes
            camarilla_s2 = prev_close - ((prev_high - prev_low) * 1.1666)
            camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
            if not np.isnan(camarilla_s2_aligned[i]) and close[i] > camarilla_s2_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves below R2 or trend changes
            camarilla_r2 = prev_close + ((prev_high - prev_low) * 1.1666)
            camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
            if not np.isnan(camarilla_r2_aligned[i]) and close[i] < camarilla_r2_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: price touches S3 in uptrend
                if daily_uptrend and close[i] <= camarilla_s3_aligned[i] * 1.001:  # Allow small buffer
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches R3 in downtrend
                elif daily_downtrend and close[i] >= camarilla_r3_aligned[i] * 0.999:  # Allow small buffer
                    position = -1
                    signals[i] = -0.25
    
    return signals