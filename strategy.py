#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend + Volume Confirmation v2
Hypothesis: Weekly pivot points (S1/S2/R1/R2) define key support/resistance zones.
In trending markets (1d EMA50 alignment), price breaks of pivot levels with volume
confirmation capture momentum moves. In ranging markets (price between S1/R1),
mean reversion at pivot levels with volume exhaustion provides counter-trend entries.
The 6h timeframe balances responsiveness with low turnover (target 15-40 trades/year).
Works in bull/bear by adapting to trend regime via 1d EMA50.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_daily_trend_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly pivot calculation (using prior week's OHLC)
    # P = (H + L + C) / 3
    # S1 = (2*P) - H, S2 = P - (H - L)
    # R1 = (2*P) - L, R2 = P + (H - L)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    s1 = (2 * pivot) - weekly_high
    s2 = pivot - (weekly_high - weekly_low)
    r1 = (2 * pivot) - weekly_low
    r2 = pivot + (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h (with shift(1) for prior week's data)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for trend filter
    ema_50_daily = df_daily['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume filter (>1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_daily_aligned[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S1 or trend reverses
            if close[i] <= s1_aligned[i] or close[i] < ema_50_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R1 or trend reverses
            if close[i] >= r1_aligned[i] or close[i] > ema_50_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine regime: trending if price > EMA50, ranging if between S1/R1
            is_trending_up = close[i] > ema_50_daily_aligned[i]
            is_trending_down = close[i] < ema_50_daily_aligned[i]
            is_ranging = (s1_aligned[i] <= close[i] <= r1_aligned[i])
            
            # Trending market: breakout of pivot levels with volume
            if is_trending_up and vol_filter[i]:
                if close[i] >= r1_aligned[i]:  # Break above R1
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= r2_aligned[i]:  # Break above R2 (stronger)
                    position = 1
                    signals[i] = 0.25
                    
            if is_trending_down and vol_filter[i]:
                if close[i] <= s1_aligned[i]:  # Break below S1
                    position = -1
                    signals[i] = -0.25
                elif close[i] <= s2_aligned[i]:  # Break below S2 (stronger)
                    position = -1
                    signals[i] = -0.25
            
            # Ranging market: mean reversion at pivot levels with volume exhaustion
            if is_ranging and vol_filter[i]:
                # Long near S1/S2 with rejection (price > open and near support)
                if (close[i] <= s1_aligned[i] * 1.005 and  # Within 0.5% of S1
                    close[i] > prices['open'].iloc[i]):    # Bullish close
                    position = 1
                    signals[i] = 0.25
                # Short near R1/R2 with rejection (price < open and near resistance)
                elif (close[i] >= r1_aligned[i] * 0.995 and  # Within 0.5% of R1
                      close[i] < prices['open'].iloc[i]):   # Bearish close
                    position = -1
                    signals[i] = -0.25
    
    return signals