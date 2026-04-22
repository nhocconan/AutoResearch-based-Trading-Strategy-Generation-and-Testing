#/usr/bin/env python3
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
    
    # Load daily data for weekly pivot calculation - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Weekly pivot uses: (Weekly High + Weekly Low + Weekly Close) / 3
    # We'll approximate with daily data using 5-day rolling window
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Weekly high, low, close (5-day rolling)
    weekly_high = pd.Series(high_daily).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_daily).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_daily).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly support and resistance levels
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_daily, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_daily, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_daily, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_daily, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_daily, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_daily, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_daily, weekly_s3)
    
    # Calculate 6h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly pivot with volatility filter
            if (close[i] > weekly_pivot_aligned[i] and 
                atr[i] > 0.01 * close[i]):  # Minimum volatility filter
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly pivot with volatility filter
            elif (close[i] < weekly_pivot_aligned[i] and 
                  atr[i] > 0.01 * close[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back through weekly pivot
            if position == 1:
                if close[i] < weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyPivot_Volatility_Filter"
timeframe = "6h"
leverage = 1.0