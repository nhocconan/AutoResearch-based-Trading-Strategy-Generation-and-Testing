#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability reversal/breakout points in ranging markets
# Strong breakout above R3 or below S3 with volume confirmation and 1d EMA34 trend alignment
# Works in both bull and bear markets by following the higher timeframe trend
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Get 1d data for Camarilla pivot calculation (using prior completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    
    # Calculate Camarilla levels for each 12h bar using prior 1d data
    camarilla_R4 = np.full(n, np.nan)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_R2 = np.full(n, np.nan)
    camarilla_R1 = np.full(n, np.nan)
    camarilla_PP = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    camarilla_S2 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_S4 = np.full(n, np.nan)
    
    for i in range(n):
        # Get the most recent completed 1d bar index
        # Since open_time is datetime64, we can compare dates directly
        if i == 0:
            continue
            
        current_date = pd.Timestamp(open_time[i]).date()
        prior_date = pd.Timestamp(open_time[i-1]).date()
        
        # Check if we've moved to a new day (UTC)
        if current_date != prior_date:
            # Find the index of the prior day's 1d bar
            # We need to get the 1d bar that closed prior to current 12h bar
            day_idx = np.sum(pd.Timestamp(df_1d.index).date < current_date)
            if day_idx > 0 and day_idx < len(high_1d):
                # Use the last completed 1d bar
                idx = day_idx - 1
                H = high_1d[idx]
                L = low_1d[idx]
                C = close_1d[idx]
                
                camarilla_PP[i] = (H + L + C) / 3
                camarilla_R1[i] = C + (H - L) * 1.1 / 12
                camarilla_R2[i] = C + (H - L) * 1.1 / 6
                camarilla_R3[i] = C + (H - L) * 1.1 / 4
                camarilla_R4[i] = C + (H - L) * 1.1 / 2
                camarilla_S1[i] = C - (H - L) * 1.1 / 12
                camarilla_S2[i] = C - (H - L) * 1.1 / 6
                camarilla_S3[i] = C - (H - L) * 1.1 / 4
                camarilla_S4[i] = C - (H - L) * 1.1 / 2
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_R3[i]) or 
            np.isnan(camarilla_S3[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 with volume spike AND 1d EMA34 uptrend
            if close[i] > camarilla_R3[i] and volume[i] > (2.0 * vol_ema_20[i]) and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S3 with volume spike AND 1d EMA34 downtrend
            elif close[i] < camarilla_S3[i] and volume[i] > (2.0 * vol_ema_20[i]) and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 or Camarilla PP
            if close[i] < camarilla_R3[i] or close[i] < camarilla_PP[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S3 or Camarilla PP
            if close[i] > camarilla_S3[i] or close[i] > camarilla_PP[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals