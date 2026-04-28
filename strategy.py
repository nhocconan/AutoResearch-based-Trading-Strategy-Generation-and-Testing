#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with volume confirmation and session filter (08-20 UTC).
# Enter long when price breaks above R1 with volume > 1.5x 20-bar average and price > 4h EMA50.
# Enter short when price breaks below S1 with volume > 1.5x 20-bar average and price < 4h EMA50.
# Exit when price returns to pivot point (PP) or opposite breakout occurs.
# Uses 4h for signal direction (trend + structure), 1h only for entry timing precision.
# Session filter reduces noise trades outside active hours.
# Discrete position sizing (0.20) controls risk. Target: 60-150 total trades over 4 years.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot and EMA calculation (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot points (based on previous day's OHLC)
    # For intraday, we use the previous 4h bar's high, low, close
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_4h + low_4h + close_4h) / 3.0
    
    # Camarilla levels
    r1 = pp + (high_4h - low_4h) * 1.1 / 12
    s1 = pp - (high_4h - low_4h) * 1.1 / 12
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 1h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Determine breakout conditions
        long_breakout = close[i] > r1_aligned[i]
        short_breakout = close[i] < s1_aligned[i]
        
        # Trend filter: price > EMA50 for long, price < EMA50 for short
        long_trend = close[i] > ema_50_aligned[i]
        short_trend = close[i] < ema_50_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and long_trend and volume_confirm[i]
        short_entry = short_breakout and short_trend and volume_confirm[i]
        
        # Exit conditions: price returns to pivot point (PP) or opposite breakout
        long_exit = close[i] < pp_aligned[i]
        short_exit = close[i] > pp_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals