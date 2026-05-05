#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1h Williams %R extreme reversal with 4h EMA20 trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold bounce) AND 4h EMA20 > previous 4h EMA20 (uptrend) AND volume > 1.8 * avg_volume(20) on 4h
# Short when Williams %R(14) crosses below -20 (overbought rejection) AND 4h EMA20 < previous 4h EMA20 (downtrend) AND volume > 1.8 * avg_volume(20) on 4h
# Exit when Williams %R crosses back through -50 (mean reversion) or opposite extreme is reached
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R captures short-term exhaustion while EMA20 filters for higher timeframe trend
# Volume confirmation (1.8x) ensures breakout validity without excessive trading
# Works in both bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets

name = "4h_1hWilliamsR_4hEMA20_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data ONCE before loop for Williams %R calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:  # Need at least 14 completed 1h bars for Williams %R
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 1h Williams %R(14): %R = (highest_high - close) / (highest_high - lowest_low) * -100
    highest_high_1h = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    lowest_low_1h = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    williams_r_1h = -100 * (highest_high_1h - close_1h) / (highest_high_1h - lowest_low_1h)
    # Handle division by zero (when high == low)
    williams_r_1h[highest_high_1h == lowest_low_1h] = -50.0
    
    # Align 1h Williams %R to 4h timeframe (wait for completed 1h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1h, williams_r_1h)
    
    # Get 4h data ONCE before loop for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 completed 4h bars for EMA20
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below), 4h EMA20 rising (uptrend), volume confirmation, in session
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above), 4h EMA20 falling (downtrend), volume confirmation, in session
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (mean reversion) or reaches overbought
            if williams_r_aligned[i] > -50 or williams_r_aligned[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (mean reversion) or reaches oversold
            if williams_r_aligned[i] < -50 or williams_r_aligned[i] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals