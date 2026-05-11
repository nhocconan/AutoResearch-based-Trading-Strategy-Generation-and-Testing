#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_DailyTrend
Hypothesis: Price often respects weekly pivot levels (R1/S1, R2/S2). When price breaks above weekly R1 with daily trend alignment (close > daily EMA50), it signals continuation. Similarly, breaks below weekly S1 with daily trend alignment (close < daily EMA50) signal continuation. Uses volume confirmation to avoid false breaks. Designed for low frequency (15-25 trades/year) to minimize fee drag while capturing major moves in both bull and bear markets.
"""

name = "6h_Weekly_Pivot_Breakout_DailyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # 6h price data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- Weekly Pivot Points (using previous week) ---
    # Previous week's OHLC
    prev_weekly_high = np.roll(df_weekly['high'].values, 1)
    prev_weekly_low = np.roll(df_weekly['low'].values, 1)
    prev_weekly_close = np.roll(df_weekly['close'].values, 1)
    # First bar initialization
    prev_weekly_high[0] = df_weekly['high'].values[0]
    prev_weekly_low[0] = df_weekly['low'].values[0]
    prev_weekly_close[0] = df_weekly['close'].values[0]
    
    # Weekly pivot calculation
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_range = prev_weekly_high - prev_weekly_low
    
    # Weekly support/resistance levels
    weekly_r1 = weekly_pivot + (weekly_range * 1.0)
    weekly_s1 = weekly_pivot - (weekly_range * 1.0)
    weekly_r2 = weekly_pivot + (weekly_range * 2.0)
    weekly_s2 = weekly_pivot - (weekly_range * 2.0)
    
    # Align weekly levels to 6h timeframe
    weekly_r1_6h = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    weekly_s2_6h = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # --- Daily Trend Filter (EMA50) ---
    daily_close = df_daily['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_6h = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    # --- Volume Confirmation (20-period average) ---
    volume_ma20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after sufficient warmup for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i]) or 
            np.isnan(daily_ema50_6h[i]) or np.isnan(volume_ma20[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR based on 6h range
                atr_est = np.abs(high_6h[i] - low_6h[i])
                if position == 1 and close_6h[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_6h[i] > 1.5 * volume_ma20[i]
        
        if position == 0:
            # Look for breakout entries with trend alignment and volume
            # Long: break above weekly R1 with close > daily EMA50 and volume
            if (close_6h[i] > weekly_r1_6h[i] and 
                close_6h[i] > daily_ema50_6h[i] and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            # Short: break below weekly S1 with close < daily EMA50 and volume
            elif (close_6h[i] < weekly_s1_6h[i] and 
                  close_6h[i] < daily_ema50_6h[i] and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit on break below weekly S1 or trend reversal
                if close_6h[i] < weekly_s1_6h[i] or close_6h[i] < daily_ema50_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit on break above weekly R1 or trend reversal
                if close_6h[i] > weekly_r1_6h[i] or close_6h[i] > daily_ema50_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals