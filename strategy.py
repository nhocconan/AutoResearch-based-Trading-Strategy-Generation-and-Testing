#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivot calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Calculate Camarilla levels for previous day (shifted by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_range = np.roll(daily_range, 1)
    
    # Set first day values to NaN since we don't have previous day data
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_range[0] = np.nan
    
    # Calculate Camarilla R3 and S3 from previous day (tighter levels for fewer trades)
    r3 = prev_close + (prev_range * 1.1 / 4)
    s3 = prev_close - (prev_range * 1.1 / 4)
    
    # Calculate 1-day EMA50 for trend filter (slower for fewer signals)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (30-period on 4h for stricter confirmation)
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > 2.5 * vol_ma30  # Require 2.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma30[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + above 1d EMA50 + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + below 1d EMA50 + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite S3/R3 level or trend changes
            if position == 1:
                if (close[i] < s3_aligned[i] or 
                    close[i] < ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > r3_aligned[i] or 
                    close[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA50_Volume_Session"
timeframe = "4h"
leverage = 1.0