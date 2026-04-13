#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla breakout with 4h volume confirmation and 1d trend filter
    # Enter long when price breaks above R4 with volume > 1.5x 20-bar avg AND 1d close > 20 EMA
    # Enter short when price breaks below S4 with volume > 1.5x 20-bar avg AND 1d close < 20 EMA
    # Exit when price crosses the 1d close (midpoint)
    # Uses 1d for Camarilla levels and trend, 4h for volume confirmation, 1h for timing
    # Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    cam_high_low = high_1d - low_1d
    camarilla_r4 = close_1d + (cam_high_low * 1.1 / 2)
    camarilla_s4 = close_1d - (cam_high_low * 1.1 / 2)
    camarilla_mid = close_1d  # midpoint is the 1d close
    
    # Align 1d Camarilla levels and EMA to 1h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Get 4h data for volume confirmation (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h 20-bar average volume
    volume_4h_series = pd.Series(volume_4h)
    avg_volume_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 4h volume average to 1h timeframe
    avg_volume_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_volume_4h)
    
    # Volume confirmation: volume > 1.5x 4h 20-bar average volume
    volume_confirmed = volume > (1.5 * avg_volume_4h_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):  # start from 20 to ensure indicators are ready
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(volume_confirmed[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > camarilla_r4_aligned[i]  # break above R4
        breakout_down = close[i] < camarilla_s4_aligned[i]  # break below S4
        
        # Trend filter: 1d close above/below 20 EMA
        uptrend = close_1d_aligned[i] > ema_20_1d_aligned[i] if 'close_1d_aligned' in locals() else close[i] > ema_20_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_20_1d_aligned[i] if 'close_1d_aligned' in locals() else close[i] < ema_20_1d_aligned[i]
        
        # Entry conditions with volume confirmation and trend filter
        long_entry = breakout_up and volume_confirmed[i] and uptrend and position != 1
        short_entry = breakout_down and volume_confirmed[i] and downtrend and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < camarilla_mid_aligned[i])
        exit_short = (position == -1 and close[i] > camarilla_mid_aligned[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4d_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0