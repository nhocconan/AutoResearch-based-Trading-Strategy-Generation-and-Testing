#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels (R3/S3, R4/S4) with 1d EMA trend filter and volume spike confirmation.
# Enter long when price breaks above weekly R4 with 1d EMA34 uptrend and volume > 2x 20-bar average.
# Enter short when price breaks below weekly S4 with 1d EMA34 downtrend and volume > 2x 20-bar average.
# Exit when price retraces to weekly R3/S3 or opposite Camarilla level is touched.
# Weekly Camarilla provides structural support/resistance from higher timeframe.
# EMA34 filter ensures alignment with daily trend to avoid counter-trend trades.
# Volume spike confirms institutional participation in breakouts.
# Discrete position sizing (0.25) limits risk. Target: 50-150 total trades over 4 years.

name = "6h_WeeklyCamarilla_R3S3_R4S4_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for EMA filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # Camarilla formula: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # where C = close, H = high, L = low of previous period
    
    # Use previous week's OHLC (shift by 1 to avoid look-ahead)
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Calculate Camarilla levels
    R4 = prev_week_close + ((prev_week_high - prev_week_low) * 1.1 / 2)
    R3 = prev_week_close + ((prev_week_high - prev_week_low) * 1.1 / 4)
    S3 = prev_week_close - ((prev_week_high - prev_week_low) * 1.1 / 4)
    S4 = prev_week_close - ((prev_week_high - prev_week_low) * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter conditions
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        # Breakout conditions
        breakout_above_R4 = close[i] > R4_aligned[i]
        breakdown_below_S4 = close[i] < S4_aligned[i]
        
        # Retracement conditions (exit)
        retrace_to_R3 = close[i] <= R3_aligned[i]
        retrace_to_S3 = close[i] >= S3_aligned[i]
        
        # Entry conditions
        long_entry = breakout_above_R4 and price_above_ema and volume_confirm[i]
        short_entry = breakdown_below_S4 and price_below_ema and volume_confirm[i]
        
        # Exit conditions
        long_exit = retrace_to_R3 or breakdown_below_S4  # Exit if retrace to R3 or break below S4
        short_exit = retrace_to_S3 or breakout_above_R4   # Exit if retrace to S3 or break above R4
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals