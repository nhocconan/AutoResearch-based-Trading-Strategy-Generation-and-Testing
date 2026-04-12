#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels from previous week
    # Previous week's high, low, close
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # First week has no previous week
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels
    # Resistance levels
    r4 = prev_close + (prev_high - prev_low) * 1.500
    r3 = prev_close + (prev_high - prev_low) * 1.250
    r2 = prev_close + (prev_high - prev_low) * 1.166
    r1 = prev_close + (prev_high - prev_low) * 1.083
    # Support levels
    s1 = prev_close - (prev_high - prev_low) * 1.083
    s2 = prev_close - (prev_high - prev_low) * 1.166
    s3 = prev_close - (prev_high - prev_low) * 1.250
    s4 = prev_close - (prev_high - prev_low) * 1.500
    
    # Align Camarilla levels to daily timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume filter: 20-day average on daily data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Trend filter: 50-day EMA on daily close
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_ok[i]) or
            np.isnan(ema_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout signals with volume confirmation
        # Long: Close breaks above R3 (strong resistance) in uptrend
        long_signal = close[i] > r3_aligned[i] and uptrend[i] and volume_ok[i]
        # Short: Close breaks below S3 (strong support) in downtrend
        short_signal = close[i] < s3_aligned[i] and downtrend[i] and volume_ok[i]
        
        # Exit when price returns to midpoint (mean reversion)
        midpoint = (r1_aligned[i] + s1_aligned[i]) / 2
        exit_long = close[i] < midpoint
        exit_short = close[i] > midpoint
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals