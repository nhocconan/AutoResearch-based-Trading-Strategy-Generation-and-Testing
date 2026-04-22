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
    
    # Load weekly data for monthly pivot - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate monthly pivot points from weekly data (using 4-week lookback)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Monthly high/low/close approximation using 4-week lookback
    monthly_high = pd.Series(high_weekly).rolling(window=4, min_periods=4).max().values
    monthly_low = pd.Series(low_weekly).rolling(window=4, min_periods=4).min().values
    monthly_close = pd.Series(close_weekly).rolling(window=4, min_periods=4).last().values
    
    # Calculate monthly pivot point: (H + L + C) / 3
    monthly_pivot = (monthly_high + monthly_low + monthly_close) / 3.0
    
    # Calculate monthly support and resistance levels
    monthly_range = monthly_high - monthly_low
    monthly_r1 = 2 * monthly_pivot - monthly_low
    monthly_s1 = 2 * monthly_pivot - monthly_high
    monthly_r2 = monthly_pivot + monthly_range
    monthly_s2 = monthly_pivot - monthly_range
    monthly_r3 = monthly_high + 2 * (monthly_pivot - monthly_low)
    monthly_s3 = monthly_low - 2 * (monthly_high - monthly_pivot)
    
    # Calculate ATR(14) from weekly data for volatility filter
    tr1 = high_weekly - low_weekly
    tr2 = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align monthly pivot levels and ATR to daily timeframe
    monthly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, monthly_pivot)
    monthly_r1_aligned = align_htf_to_ltf(prices, df_weekly, monthly_r1)
    monthly_s1_aligned = align_htf_to_ltf(prices, df_weekly, monthly_s1)
    monthly_r2_aligned = align_htf_to_ltf(prices, df_weekly, monthly_r2)
    monthly_s2_aligned = align_htf_to_ltf(prices, df_weekly, monthly_s2)
    monthly_r3_aligned = align_htf_to_ltf(prices, df_weekly, monthly_r3)
    monthly_s3_aligned = align_htf_to_ltf(prices, df_weekly, monthly_s3)
    atr_14_aligned = align_htf_to_ltf(prices, df_weekly, atr_14)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(monthly_pivot_aligned[i]) or np.isnan(monthly_r1_aligned[i]) or 
            np.isnan(monthly_s1_aligned[i]) or np.isnan(monthly_r2_aligned[i]) or
            np.isnan(monthly_s2_aligned[i]) or np.isnan(monthly_r3_aligned[i]) or
            np.isnan(monthly_s3_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price closes above monthly R1 with volume confirmation and sufficient volatility
            if (close[i] > monthly_r1_aligned[i] and 
                volume[i] > 1.8 * vol_avg_20[i] and
                atr_14_aligned[i] > 0.5 * np.mean(atr_14_aligned[max(0, i-50):i+1])):  # Volatility filter
                signals[i] = 0.25
                position = 1
            # Short: Price closes below monthly S1 with volume confirmation and sufficient volatility
            elif (close[i] < monthly_s1_aligned[i] and 
                  volume[i] > 1.8 * vol_avg_20[i] and
                  atr_14_aligned[i] > 0.5 * np.mean(atr_14_aligned[max(0, i-50):i+1])):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to monthly pivot level
            if position == 1:
                if close[i] < monthly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > monthly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_MonthlyPivot_R1S1_Volume_ATR_Filter_v1"
timeframe = "1d"
leverage = 1.0