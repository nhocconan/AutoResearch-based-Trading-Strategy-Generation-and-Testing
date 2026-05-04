#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Designed for 1h timeframe to target 15-37 trades/year (60-150 total over 4 years)
# Uses 4h for signal direction (trend and structure) and 1h only for entry timing precision
# Includes session filter (08-20 UTC) to reduce noise trades
# Fixed position size of 0.20 to control drawdown and minimize fee churn
# Works in bull markets by buying breakouts above R1 in uptrends
# Works in bear markets by selling breakdowns below S1 in downtrends
# Avoids false breakouts via trend and volume filters

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_Session"
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
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter and camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate camarilla levels: R1, S1 from 4h OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_range = high_4h - low_4h
    r1 = close_4h + 1.1 * camarilla_range / 4
    s1 = close_4h - 1.1 * camarilla_range / 4
    
    # Align camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: 2.0x 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: close breaks above R1 + volume confirmation + price above 4h EMA50 (uptrend)
            if (close[i] > r1_aligned[i] and volume_confirmed and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: close breaks below S1 + volume confirmation + price below 4h EMA50 (downtrend)
            elif (close[i] < s1_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price falls below S1 (mean reversion) OR below 4h EMA50 (trend change)
            if close[i] < s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price rises above R1 (mean reversion) OR above 4h EMA50 (trend change)
            if close[i] > r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals