#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R3S3_Breakout_Trend_v1
Hypothesis: Uses weekly Camarilla pivot levels (R3/S3) for breakout signals on the daily chart.
Trades only in bullish weekly trends (price > weekly EMA34) to avoid bearish whipsaws.
Requires volume confirmation (volume > 1.5x 20-day average) to filter false breakouts.
Designed for low trade frequency (~10-20 trades/year) with strong directional bias.
Works in bull markets via breakouts and avoids bear markets via trend filter.
"""

name = "1d_Weekly_Camarilla_R3S3_Breakout_Trend_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla pivots and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA34 for trend filter ---
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # --- Weekly Camarilla levels (R3, S3) ---
    # Using previous week's OHLC to calculate current week's levels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close_prev = df_weekly['close'].values
    
    # Calculate Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    rng = weekly_high - weekly_low
    r3 = weekly_close_prev + rng * 1.1 / 4
    s3 = weekly_close_prev - rng * 1.1 / 4
    
    # Align to daily (using previous week's levels for current week's trading)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # --- Volume confirmation (daily) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema34_weekly_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: only trade long in bullish weekly trend
        bullish_trend = close[i] > ema34_weekly_aligned[i]
        
        if position == 0:
            # Look for breakout above R3 with volume confirmation
            if bullish_trend and close[i] > r3_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
        else:
            # Exit conditions: close below S3 or trend turns bearish
            if position == 1:
                exit_signal = (close[i] < s3_aligned[i]) or (close[i] <= ema34_weekly_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
    
    return signals