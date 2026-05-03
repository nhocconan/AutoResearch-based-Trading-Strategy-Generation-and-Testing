#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla levels provide high-probability intraday reversal points; breakouts beyond R3/S3
# with volume spike and aligned weekly trend offer strong continuation moves. Designed for very low
# trade frequency (12-37/year) on 12h timeframe to minimize fee drag. Works in both bull and bear
# markets by trading breakouts in the direction of the higher timeframe trend.

name = "12h_Camarilla_R3S3_1wEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for trend filter and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1w volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1w['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1w indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar (HLC of previous day)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We use prior day's HLC to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior day's values (already completed)
    high_1d_lag = np.roll(high_1d, 1)
    low_1d_lag = np.roll(low_1d, 1)
    close_1d_lag = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last) - set to NaN
    high_1d_lag[0] = np.nan
    low_1d_lag[0] = np.nan
    close_1d_lag[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_range = high_1d_lag - low_1d_lag
    r3 = close_1d_lag + camarilla_range * 1.1 / 4
    s3 = close_1d_lag - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after warmup for rolled values
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in uptrend
            if close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and ema_34_aligned[i] > close[i] and volume_spike_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 with volume spike in downtrend
            elif close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and ema_34_aligned[i] < close[i] and volume_spike_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price re-enters below R3
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price re-enters above S3
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals