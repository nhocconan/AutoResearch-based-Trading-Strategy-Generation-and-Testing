#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(50) trend filter and volume confirmation
# Long when price breaks above 1h Camarilla R3 + volume spike + price > 4h EMA(50)
# Short when price breaks below 1h Camarilla S3 + volume spike + price < 4h EMA(50)
# Uses 4h EMA(50) for trend alignment to reduce whipsaw, 1d for session filter (08-20 UTC)
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Designed for low trade frequency (15-37/year on 1h) to minimize fee drag
# Works in bull (breakouts with trend) and bear (breakdowns with trend) markets
# Camarilla levels calculated from prior 1h session to avoid look-ahead

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe (wait for completed 4h bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for session filter (08-20 UTC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate session hours from 1d data (each 1d bar = 24h)
    # We'll use the date to determine if we're in 08-20 UTC session
    session_hours = pd.DatetimeIndex(df_1d['open_time']).hour.values
    session_hours_aligned = align_htf_to_ltf(prices, df_1d, session_hours)
    
    # Calculate Camarilla levels from prior 1h session (using 1h data)
    # Camarilla R3 = close + 1.1*(high-low)*1.1/4
    # Camarilla S3 = close - 1.1*(high-low)*1.1/4
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    rng = (high_series - low_series)
    camarilla_r3 = close_series + 1.1 * rng * 1.1 / 4
    camarilla_s3 = close_series - 1.1 * rng * 1.1 / 4
    # Shift by 1 to use prior bar's levels (avoid look-ahead)
    camarilla_r3_shifted = camarilla_r3.shift(1).values
    camarilla_s3_shifted = camarilla_s3.shift(1).values
    
    # Volume confirmation (1.5x 20-period average) on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20)  # 4h EMA(50), Camarilla(20), volume MA(20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_shifted[i]) or 
            np.isnan(camarilla_s3_shifted[i]) or np.isnan(volume_spike[i]) or
            np.isnan(session_hours_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = session_hours_aligned[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + volume spike + price > 4h EMA(50)
            if (close[i] > camarilla_r3_shifted[i] and volume_spike[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla S3 + volume spike + price < 4h EMA(50)
            elif (close[i] < camarilla_s3_shifted[i] and volume_spike[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price below Camarilla S3 or price below 4h EMA(50)
            if (close[i] < camarilla_s3_shifted[i] or close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price above Camarilla R3 or price above 4h EMA(50)
            if (close[i] > camarilla_r3_shifted[i] or close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals