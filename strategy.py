# 12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: Uses Camarilla R3/S3 levels on daily chart for breakout signals with weekly trend filter and volume confirmation.
# Works in bull markets via breakouts above R3 and in bear markets via breakdowns below S3.
# Weekly trend filter avoids counter-trend trades. Volume confirmation ensures institutional participation.
# Target: 15-25 trades/year to minimize fee drag while capturing strong moves.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla Levels (using previous day's OHLC) ---
    # Calculate Camarilla levels from previous day's data
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Typical price for pivot calculation
    typical_price = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R3 = typical_price + (range_val * 1.1 / 2)
    S3 = typical_price - (range_val * 1.1 / 2)
    R4 = typical_price + (range_val * 1.1)
    S4 = typical_price - (range_val * 1.1)
    
    # Align daily levels to 12h timeframe (use previous day's levels for current day)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # --- Weekly Trend Filter (50-period EMA) ---
    weekly_close = df_1w['close'].values
    ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Significant volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        price_above_weekly_ema = close[i] > ema_50_aligned[i]
        
        # Breakout signals
        long_breakout = (high[i] > R3_aligned[i]) and vol_spike[i]
        short_breakout = (low[i] < S3_aligned[i]) and vol_spike[i]
        
        if position == 0:
            # Only take longs in uptrend, shorts in downtrend
            if price_above_weekly_ema and long_breakout:
                signals[i] = 0.25
                position = 1
            elif not price_above_weekly_ema and short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches R4 or weekly trend turns bearish
                exit_signal = (high[i] > R4_aligned[i]) or (close[i] < ema_50_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches S4 or weekly trend turns bullish
                exit_signal = (low[i] < S4_aligned[i]) or (close[i] > ema_50_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3