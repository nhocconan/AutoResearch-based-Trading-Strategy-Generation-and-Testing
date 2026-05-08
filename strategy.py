#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day volume-weighted average price (VWAP) with 1-week trend filter.
# VWAP acts as dynamic support/resistance. Long when price pulls back to VWAP in uptrend with volume confirmation.
# Short when price bounces from VWAP in downtrend with volume confirmation.
# Uses 1-week trend filter to ensure alignment with higher timeframe momentum.
# Designed for low trade frequency (20-50/year) to minimize fee drag and capture high-probability mean reversion.

name = "4h_VWAP_Pullback_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    
    # Align VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_trend_up = ema_21_1w[1:] > ema_21_1w[:-1]  # Rising weekly EMA
    weekly_trend_up = np.concatenate([[False], weekly_trend_up])  # Align with daily index
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # 4h EMA(50) for intermediate trend and dynamic support/resistance
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.8x 30-period EMA
    vol_ema = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_confirm = volume > (vol_ema * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: pullback to VWAP in uptrend with volume
            if (weekly_trend_aligned[i] > 0.5 and  # Weekly uptrend
                close[i] > ema_50[i] and             # Above intermediate EMA
                close[i] <= vwap_aligned[i] * 1.005 and  # Near VWAP (allow 0.5% slack)
                close[i] >= vwap_aligned[i] * 0.995 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: bounce from VWAP in downtrend with volume
            elif (weekly_trend_aligned[i] <= 0.5 and  # Weekly downtrend
                  close[i] < ema_50[i] and            # Below intermediate EMA
                  close[i] >= vwap_aligned[i] * 0.995 and  # Near VWAP
                  close[i] <= vwap_aligned[i] * 1.005 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below VWAP or trend turns down
            if close[i] < vwap_aligned[i] * 0.995 or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above VWAP or trend turns up
            if close[i] > vwap_aligned[i] * 1.005 or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals