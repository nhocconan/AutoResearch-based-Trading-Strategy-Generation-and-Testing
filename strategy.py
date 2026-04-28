# 1d_Camarilla_Pivot_R3_S3_Breakout_WeeklyTrend_Volume_Session
# Hypothesis: On daily timeframe, Camarilla R3/S3 levels act as strong support/resistance. 
# Breakouts above R3 (with weekly uptrend and volume) signal long; breakdowns below S3 (with weekly downtrend and volume) signal short.
# Uses weekly EMA20 as trend filter and volume > 20-day average for confirmation. 
# Session filter (8-20 UTC) avoids low-liquidity hours. 
# Targets 20-40 trades/year to minimize fee drag. Works in bull (breaks higher) and bear (breaks lower) regimes.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Camarilla formulas
    range_d = high_d - low_d
    pivot_d = (high_d + low_d + close_d) / 3
    r3_d = close_d + range_d * 1.1 / 2
    s3_d = close_d - range_d * 1.1 / 2
    r1_d = close_d + range_d * 1.1 / 12
    s1_d = close_d - range_d * 1.1 / 12
    
    # Align to daily timeframe (no additional delay needed for pivot levels)
    r3_d_aligned = align_htf_to_ltf(prices, df_1d, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_1d, s3_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_d_aligned[i]) or np.isnan(s3_d_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        # Entry conditions: 
        # Long: break above daily R3 with upward trend and volume
        # Short: break below daily S3 with downward trend and volume
        long_breakout = close[i] > r3_d_aligned[i]
        short_breakout = close[i] < s3_d_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_up
        short_entry = short_breakout and vol_filter and trend_down
        
        # Exit conditions: opposite R1/S1 level touch
        long_exit = (close[i] < s1_d_aligned[i]) and position == 1
        short_exit = (close[i] > r1_d_aligned[i]) and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
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

name = "1d_Camarilla_Pivot_R3_S3_Breakout_WeeklyTrend_Volume_Session"
timeframe = "1d"
leverage = 1.0