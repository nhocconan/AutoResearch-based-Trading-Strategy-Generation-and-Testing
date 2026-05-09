# 4h_Camarilla_R3_S3_12hEMA50_Trend_Volume
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Combines precise support/resistance from Camarilla levels with trend filter from 12h EMA50
# and volume confirmation to reduce false breakouts. Designed for 4h timeframe with
# target of 75-200 trades over 4 years (19-50/year). Works in bull/bear markets by
# requiring trend alignment and volume confirmation.
name = "4h_Camarilla_R3_S3_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d
    # Using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    R4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and Camarilla calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(R4_4h[i]) or np.isnan(S4_4h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > R3_4h[i-1]  # Break above R3
        short_breakout = close[i] < S3_4h[i-1]  # Break below S3
        
        trend_up = close[i] > ema_50_4h[i]
        trend_down = close[i] < ema_50_4h[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if long_breakout and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif short_breakout and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout below S3 or trend reversal
            if close[i] < S3_4h[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout above R3 or trend reversal
            if close[i] > R3_4h[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals