#!/usr/bin/env python3
# 1d_weekly_pivot_reversion_v1
# Hypothesis: In bear/ranging markets (2025+), price tends to revert to weekly pivot levels.
# Long when: price touches weekly S1 support with bullish engulfing candle and volume > 1.5x average.
# Short when: price touches weekly R1 resistance with bearish engulfing candle and volume > 1.5x average.
# Exit when price moves back toward weekly pivot (PP) or shows opposite rejection.
# Uses weekly pivot points (PP, R1, S1) calculated from prior week's OHLC.
# Target: 15-25 trades/year to avoid fee drag in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Bullish/bearish engulfing detection
    bullish_engulf = np.full(n, False)
    bearish_engulf = np.full(n, False)
    for i in range(1, n):
        bullish_engulf[i] = (close[i] > open_price[i] and 
                            open_price[i] < close[i-1] and 
                            close[i] > open_price[i-1])
        bearish_engulf[i] = (close[i] < open_price[i] and 
                            open_price[i] > close[i-1] and 
                            close[i] < open_price[i-1])
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivot levels to daily timeframe (with 1-week delay for completion)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = vol_ma_period  # Wait for volume MA to be ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price moves back toward weekly pivot or shows bearish rejection
            if close[i] >= pp_aligned[i] or bearish_engulf[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves back toward weekly pivot or shows bullish rejection
            if close[i] <= pp_aligned[i] or bullish_engulf[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S1 support with bullish engulfing and volume surge
            if (abs(low[i] - s1_aligned[i]) < 0.005 * s1_aligned[i] and  # Within 0.5% of S1
                bullish_engulf[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R1 resistance with bearish engulfing and volume surge
            elif (abs(high[i] - r1_aligned[i]) < 0.005 * r1_aligned[i] and  # Within 0.5% of R1
                  bearish_engulf[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals