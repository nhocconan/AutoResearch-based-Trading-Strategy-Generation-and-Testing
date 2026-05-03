#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and session filter (08-20 UTC).
# Long: Close > R3 AND 4h EMA50 uptrend AND volume > 1.5x 20-period MA AND session 08-20 UTC
# Short: Close < S3 AND 4h EMA50 downtrend AND volume > 1.5x 20-period MA AND session 08-20 UTC
# Exit: Opposite Camarilla level breach or trend reversal.
# Uses Camarilla pivot structure for institutional levels, 4h EMA50 for higher timeframe trend,
# volume confirmation to reduce false signals, session filter to avoid low-liquidity hours.
# Designed for 1h timeframe with tight entry conditions to limit trades to 60-150 over 4 years.

name = "1h_Camarilla_R3S3_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for trend filter and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots from previous 4h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    hl_range_4h = df_4h['high'].values - df_4h['low'].values
    close_4h = df_4h['close'].values
    r3_4h = close_4h + 1.1 * hl_range_4h
    s3_4h = close_4h - 1.1 * hl_range_4h
    
    # Align Camarilla levels to 1h timeframe (use previous completed 4h bar)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume regime: current 1h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or 
            np.isnan(s3_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r3_level = r3_4h_aligned[i]
        s3_level = s3_4h_aligned[i]
        vol_spike = volume_spike[i]
        in_session = session_filter[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Close > R3 AND uptrend AND volume spike AND session
            if close_val > r3_level and is_uptrend and vol_spike and in_session:
                signals[i] = 0.20
                position = 1
            # Short: Close < S3 AND downtrend AND volume spike AND session
            elif close_val < s3_level and is_downtrend and vol_spike and in_session:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close < S3 (breakdown) OR trend turns down
            if close_val < s3_level or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close > R3 (breakout) OR trend turns up
            if close_val > r3_level or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals