# 1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
# Hypothesis: Camarilla R1/S1 breakout with weekly trend filter and volume confirmation
# Uses daily price action with weekly trend filter to work in both bull and bear markets
# Target: 10-30 trades/year on daily timeframe to minimize fee drag
# Weekly trend filter prevents counter-trend trades in strong trends
# Volume confirmation ensures institutional participation
# Focus on R1/S1 levels (inner levels) for higher probability reversals/breakouts

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
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
    
    # Get daily data for Camarilla pivot calculation (same timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels - focus on R1/S1 (inner levels)
    R1 = prev_close + 1.1 * prev_range / 12
    S1 = prev_close - 1.1 * prev_range / 12
    R2 = prev_close + 1.1 * prev_range / 6
    S2 = prev_close - 1.1 * prev_range / 6
    
    # Align to daily timeframe (no alignment needed as same timeframe)
    R1_1d = R1
    S1_1d = S1
    R2_1d = R2
    S2_1d = S2
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA20 for trend direction
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume filter: above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_1d[i]) or np.isnan(S1_1d[i]) or 
            np.isnan(R2_1d[i]) or np.isnan(S2_1d[i]) or 
            np.isnan(weekly_ema_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above R2 with weekly uptrend
            if (close[i] > R2_1d[i] and 
                close[i] > weekly_ema_1d[i] and  # weekly uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S2 with weekly downtrend
            elif (close[i] < S2_1d[i] and 
                  close[i] < weekly_ema_1d[i] and  # weekly downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below R1 (mean reversion to inner level)
            if close[i] < R1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S1 (mean reversion to inner level)
            if close[i] > S1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals