# 1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2
# Hypothesis: Uses weekly Camarilla R1/S1 levels from previous week, weekly EMA200 trend filter, and daily volume spike for confirmation.
# Works in bull (breakouts with trend) and bear (mean reversion at extreme levels with volume confirmation).
# Target: 30-100 trades over 4 years (7-25/year) to avoid fee drag.
# Timeframe: 1d (primary), HTF: 1w for trend and levels.
# Edge: Weekly structure + volume confirmation reduces false breaks; EMA200 filter avoids counter-trend trades.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R1, S1) from previous week
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Camarilla formulas for R1 and S1 (inner levels)
    R1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    S1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # Align Camarilla levels to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Calculate 1w EMA200 for trend filter
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get daily data for volume spike (20-period average)
    vol_1d = df_1w['volume'].values  # Use weekly volume for weekly context
    vol_avg_20 = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i >= 19:
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_avg_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current week's data (last completed week)
        idx_1w = 0
        while idx_1w < len(df_1w) and df_1w.iloc[idx_1w]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1w += 1
        idx_1w -= 1  # last completed week
        
        if idx_1w < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_current = df_1w['volume'].iloc[idx_1w]
        vol_avg_20_current = vol_avg_20[idx_1w]
        
        if np.isnan(vol_current) or np.isnan(vol_avg_20_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current weekly volume > 2.0x 20-period average
        vol_confirmed = vol_current > 2.0 * vol_avg_20_current
        
        # Current price
        price = close[i]
        R1_level = R1_aligned[i]
        S1_level = S1_aligned[i]
        ema200 = ema200_1w_aligned[i]
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                # Long when price breaks above R1 and above weekly EMA200 (bullish alignment)
                if price > R1_level and price > ema200:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks below S1 and below weekly EMA200 (bearish alignment)
                elif price < S1_level and price < ema200:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Manage long position
            exit_signal = False
            # Exit when price breaks below S1 or volume confirmation lost
            if price < S1_level:
                exit_signal = True
            elif not vol_confirmed:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Manage short position
            exit_signal = False
            # Exit when price breaks above R1 or volume confirmation lost
            if price > R1_level:
                exit_signal = True
            elif not vol_confirmed:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals