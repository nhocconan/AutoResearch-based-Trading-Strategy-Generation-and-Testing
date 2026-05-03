#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 in 4h uptrend with volume spike (>1.5x 20-period volume MA).
# Short when price breaks below Camarilla S3 in 4h downtrend with volume spike.
# Camarilla pivots provide precise intraday support/resistance levels that work well in ranging and trending markets.
# 4h EMA34 ensures higher timeframe alignment, avoiding counter-trend trades.
# Volume spike confirms institutional participation. Designed for 1h timeframe to achieve 60-150 total trades over 4 years (15-37/year).
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.

name = "1h_Camarilla_R3S3_4hEMA34_VolumeSpike"
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
    
    # Pre-compute session hours for filtering (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + range_1d * 1.1 / 4  # R3 level
    camarilla_s3 = close_1d - range_1d * 1.1 / 4  # S3 level
    
    # Align Camarilla levels to lower timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside trading session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_34_4h_aligned[i]   # 4h uptrend
        trend_down = close_val < ema_34_4h_aligned[i]  # 4h downtrend
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 4h uptrend AND volume spike
            if close_val > r3_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND 4h downtrend AND volume spike
            elif close_val < s3_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price breaks below Camarilla S3 (reversal signal)
            if close_val < s3_aligned[i]:
                exit_signal = True
            # Exit: 4h trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price breaks above Camarilla R3 (reversal signal)
            if close_val > r3_aligned[i]:
                exit_signal = True
            # Exit: 4h trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals