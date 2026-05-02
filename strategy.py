#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation
# Uses 1h timeframe for precise entry timing, with 4h EMA34 for intermediate trend direction and 1d Camarilla levels for key support/resistance.
# Volume confirmation (2.0x 20-period average) ensures breakouts have conviction.
# Designed for 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend) by only taking trades in direction of 4h EMA34.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate prior day's Camarilla levels (using 1d data)
    # Camarilla: based on prior day's high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prior_high = df_1d['high'].shift(1).values  # prior day's high
    prior_low = df_1d['low'].shift(1).values    # prior day's low
    prior_close = df_1d['close'].shift(1).values # prior day's close
    
    # Calculate Camarilla levels (R3/S3 are significant breakout levels)
    R3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    S3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for prior day to complete)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: 2.0x 20-period average (20*1h = ~20 hours)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA34 and Camarilla)
    start_idx = max(34, 30)  # 34 bars for EMA34, 30 bars to ensure prior day data available
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike AND price > 4h EMA34 (bullish trend)
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with volume spike AND price < 4h EMA34 (bearish trend)
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below R3 (failed breakout) OR price below 4h EMA34 (trend change)
            if close[i] < R3_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (failed breakdown) OR price above 4h EMA34 (trend change)
            if close[i] > S3_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals