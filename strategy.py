#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Long when price is above Alligator lips (green line) in 1w uptrend with volume spike (>2.0x 20-period volume MA).
# Short when price is below Alligator lips (green line) in 1w downtrend with volume spike.
# Williams Alligator: Jaw (blue) = SMA(13,8), Teeth (red) = SMA(8,5), Lips (green) = SMA(5,3)
# Uses 1w EMA50 for higher timeframe trend alignment to avoid counter-trend trades.
# Volume spike confirms institutional participation. Designed for 1d timeframe to achieve 30-100 total trades over 4 years.

name = "1d_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams Alligator calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Williams Alligator from 1w data
    close_1w = df_1w['close'].values
    
    # Alligator lines: Jaw (SMA 13, 8), Teeth (SMA 8, 5), Lips (SMA 5, 3)
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to lower timeframe (1w -> 1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price above Alligator lips AND 1w uptrend AND volume spike
            if close_val > lips_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price below Alligator lips AND 1w downtrend AND volume spike
            elif close_val < lips_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price below Alligator lips (trend change)
            if close_val < lips_aligned[i]:
                exit_signal = True
            # Exit: 1w trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price above Alligator lips (trend change)
            if close_val > lips_aligned[i]:
                exit_signal = True
            # Exit: 1w trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals