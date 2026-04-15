#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context (1d)
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily ATR for volatility filter
    daily_tr1 = daily_high[1:] - daily_low[1:]
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    daily_tr = np.concatenate([[np.nan], np.maximum(daily_tr1, np.maximum(tr2, tr3))])
    daily_atr = pd.Series(daily_tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA40 for trend filter
    daily_close_series = pd.Series(daily_close)
    daily_ema40 = daily_close_series.ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Calculate daily pivot points (standard)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    daily_r1 = daily_close + daily_range * 1.1 / 12
    daily_s1 = daily_close - daily_range * 1.1 / 12
    daily_r2 = daily_close + daily_range * 1.1 / 6
    daily_s2 = daily_close - daily_range * 1.1 / 6
    daily_r3 = daily_close + daily_range * 1.1 / 4
    daily_s3 = daily_close - daily_range * 1.1 / 4
    daily_r4 = daily_close + daily_range * 1.1 / 2
    daily_s4 = daily_close - daily_range * 1.1 / 2
    
    # Align daily data to 12h timeframe (wait for daily close)
    daily_pivot_12h = align_htf_to_ltf(prices, daily, daily_pivot)
    daily_r1_12h = align_htf_to_ltf(prices, daily, daily_r1)
    daily_s1_12h = align_htf_to_ltf(prices, daily, daily_s1)
    daily_r2_12h = align_htf_to_ltf(prices, daily, daily_r2)
    daily_s2_12h = align_htf_to_ltf(prices, daily, daily_s2)
    daily_r3_12h = align_htf_to_ltf(prices, daily, daily_r3)
    daily_s3_12h = align_htf_to_ltf(prices, daily, daily_s3)
    daily_r4_12h = align_htf_to_ltf(prices, daily, daily_r4)
    daily_s4_12h = align_htf_to_ltf(prices, daily, daily_s4)
    daily_ema40_12h = align_htf_to_ltf(prices, daily, daily_ema40)
    daily_atr_12h = align_htf_to_ltf(prices, daily, daily_atr)
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(300, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_pivot_12h[i]) or np.isnan(daily_r1_12h[i]) or 
            np.isnan(daily_s1_12h[i]) or np.isnan(daily_r2_12h[i]) or 
            np.isnan(daily_s2_12h[i]) or np.isnan(daily_r3_12h[i]) or 
            np.isnan(daily_s3_12h[i]) or np.isnan(daily_r4_12h[i]) or 
            np.isnan(daily_s4_12h[i]) or np.isnan(daily_ema40_12h[i]) or 
            np.isnan(daily_atr_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Long: close above R3 and above daily EMA40 (trend + breakout)
            if close[i] > daily_r3_12h[i] and close[i] > daily_ema40_12h[i]:
                signals[i] = 0.25
            # Short: close below S3 and below daily EMA40 (trend + breakout)
            elif close[i] < daily_s3_12h[i] and close[i] < daily_ema40_12h[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Flat if conditions not met
        else:
            signals[i] = 0.0  # Flat if volume filter fails
    
    return signals

name = "12h_Pivot_R3_S3_EMA40_Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0