#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly trend filter and volume confirmation.
# Camarilla levels provide clear support/resistance based on prior day's range.
# Weekly trend filter ensures we only trade in direction of higher timeframe momentum.
# Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (breakouts above R3/R4 in uptrend) and bear markets (breakouts below S3/S4 in downtrend).
name = "6h_Camarilla_R3S3_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_high = np.roll(high_d, 1)
    prev_low = np.roll(low_d, 1)
    prev_close = np.roll(close_d, 1)
    # First value will be invalid (rolled from last), set to nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    R4 = prev_close + rang * 1.1 / 2
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4
    S4 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Weekly trend filter: EMA34 slope
    weekly_close = df_1w['close'].values
    ema_34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = np.diff(ema_34, prepend=np.nan)
    ema_34_slope_6h = align_htf_to_ltf(prices, df_1w, ema_34_slope)
    
    # Volume confirmation: 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or
            np.isnan(ema_34_slope_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 AND weekly uptrend AND volume confirmation
            long_breakout = close[i] > R3_6h[i]
            weekly_uptrend = ema_34_slope_6h[i] > 0
            if vol_confirm and weekly_uptrend and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND weekly downtrend AND volume confirmation
            elif vol_confirm and (ema_34_slope_6h[i] < 0) and close[i] < S3_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S3 OR weekly trend turns down
            exit_condition = close[i] < S3_6h[i] or ema_34_slope_6h[i] < 0
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R3 OR weekly trend turns up
            exit_condition = close[i] > R3_6h[i] or ema_34_slope_6h[i] > 0
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals