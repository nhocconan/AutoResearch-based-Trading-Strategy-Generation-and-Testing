# 6h Camarilla R3/S3 Breakout with 12h Trend and Volume Spike
# Hypothesis: Camarilla R3/S3 levels from daily pivots act as strong support/resistance zones. 
# Breaking above R3 or below S3 with volume confirmation and aligned 12h trend captures momentum moves.
# Works in bull/bear: In bull markets, breaks above R3 continue up; in bear markets, breaks below S3 continue down.
# Uses 6h timeframe with 12h trend filter to reduce whipsaw and increase win rate.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee limits.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R4 = close + 1.1*(high-low)*1.5, R3 = close + 1.1*(high-low)*1.25
    # S3 = close - 1.1*(high-low)*1.25, S4 = close - 1.1*(high-low)*1.5
    # We only need R3 and S3 for entry
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range * 1.25
    s3 = close_1d - 1.1 * camarilla_range * 1.25
    
    # Align daily Camarilla levels to 6h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume confirmation: current volume > 2.0x average volume (to filter weak breakouts)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 12h EMA
        uptrend = close[i] > ema_20_12h_aligned[i]
        downtrend = close[i] < ema_20_12h_aligned[i]
        
        # Breakout conditions at Camarilla R3/S3
        breakout_up = close[i] > r3_aligned[i]
        breakout_down = close[i] < s3_aligned[i]
        
        # Entry conditions: require trend + breakout + volume confirmation
        long_entry = uptrend and breakout_up and volume_confirm[i]
        short_entry = downtrend and breakout_down and volume_confirm[i]
        
        # Exit conditions: when trend reverses or opposite Camarilla level break
        if position == 1:
            exit_condition = not uptrend or close[i] < s3_aligned[i]  # Exit if trend fails or breaks S3
        elif position == -1:
            exit_condition = not downtrend or close[i] > r3_aligned[i]  # Exit if trend fails or breaks R3
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
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

name = "6h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0