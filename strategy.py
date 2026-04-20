#!/usr/bin/env python3
# 1d_1W_Pivot_R3S3_Volume_Confirmation
# Hypothesis: Daily pivot S3/R3 bounce with weekly trend filter and volume confirmation.
# Trades only in direction of weekly trend to avoid counter-trend trades.
# Target: 20-60 trades over 4 years (5-15/year). Works in bull/bear via weekly trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1W_Pivot_R3S3_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend and pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly trend: EMA34 > EMA89 = uptrend, else downtrend ===
    close_1w = df_1w['close'].values
    ema89_1w = pd.Series(close_1w).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_uptrend = ema34_1w > ema89_1w  # True for uptrend, False for downtrend
    
    # === Daily pivot levels (R3, S3) from prior day ===
    high_1d = df_1w['high'].values  # Using weekly high/low/close for pivot? No - need daily
    # Actually need daily data for pivot - get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point for prior day (use previous day's data)
    # Shift by 1 to use prior day's OHLC
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    prior_high[0] = np.nan  # First day has no prior
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    pivot_1d = (prior_high + prior_low + prior_close) / 3.0
    range_1d = prior_high - prior_low
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_1d = prior_close + (range_1d * 1.1 / 4)
    s3_1d = prior_close - (range_1d * 1.1 / 4)
    
    # === Daily volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all weekly and daily levels to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        weekly_up = weekly_uptrend_aligned[i]
        r3_1d_val = r3_1d_aligned[i]
        s3_1d_val = s3_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(weekly_up) or np.isnan(r3_1d_val) or np.isnan(s3_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price rejects S3 (bounces off support) with volume confirmation, only in weekly uptrend
            if weekly_up and (close_val < s3_1d_val and  # Price touched or went below S3
                            prices['low'].iloc[i] <= s3_1d_val and  # Confirmed touch of S3
                            close_val > s3_1d_val and  # Now bouncing back above S3
                            vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R3 (bounces off resistance) with volume confirmation, only in weekly downtrend
            elif (not weekly_up) and (close_val > r3_1d_val and  # Price touched or went above R3
                            prices['high'].iloc[i] >= r3_1d_val and  # Confirmed touch of R3
                            close_val < r3_1d_val and  # Now falling back below R3
                            vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price reaches R3 or shows weakness
            if close_val >= r3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price reaches S3 or shows weakness
            if close_val <= s3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals