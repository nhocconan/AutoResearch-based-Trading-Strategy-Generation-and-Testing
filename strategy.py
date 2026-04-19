#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 breakout with daily volume confirmation and weekly trend filter.
# Long when price breaks above R1 AND volume > 1.5x daily average volume AND weekly close > weekly SMA(50)
# Short when price breaks below S1 AND volume > 1.5x daily average volume AND weekly close < weekly SMA(50)
# Exit when price crosses back through the Camarilla central pivot point
# Uses Camarilla for precise intraday levels, volume for confirmation, weekly trend filter to avoid counter-trend trades.
# Target: 20-30 trades/year per symbol.

name = "4h_Camarilla_R1S1_Volume_WeeklyTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12, PP = (H+L+C)/3
    # We need previous day's HLC to calculate today's levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r1 = camarilla_pp + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = camarilla_pp - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 4h timeframe (already delayed by shift(1) for previous day)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get daily average volume for confirmation (20-day average)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get weekly trend filter: weekly close vs weekly SMA(50)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 100)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(weekly_sma50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp = camarilla_pp_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        weekly_sma50_val = weekly_sma50_aligned[i]
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_trend_up = weekly_close[i] > weekly_sma50_val if not np.isnan(weekly_close[i]) else False
        weekly_trend_down = weekly_close[i] < weekly_sma50_val if not np.isnan(weekly_close[i]) else False
        
        if position == 0:
            # Long entry: break above R1 + volume spike + weekly uptrend
            if price > r1 and vol > 1.5 * vol_ma and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + weekly downtrend
            elif price < s1 and vol > 1.5 * vol_ma and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below central pivot point
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above central pivot point
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals