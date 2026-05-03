#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level in 1d uptrend with volume spike (>1.8x 20-period volume MA).
# Short when price breaks below Camarilla S3 level in 1d downtrend with volume spike.
# Camarilla levels provide high-probability reversal/breakout zones based on prior day's range.
# 1d EMA34 ensures higher timeframe alignment, avoiding counter-trend trades in bear markets.
# Volume spike confirms institutional participation. Designed for 4h timeframe to achieve 75-200 total trades over 4 years.
# This strategy focuses on BTC/ETH by requiring 1d trend alignment, reducing SOL-only bias.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h data using prior 4h bar's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Prior bar values (shifted by 1 to avoid look-ahead)
    high_prev = np.roll(high_4h, 1)
    low_prev = np.roll(low_4h, 1)
    close_prev = np.roll(close_4h, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Camarilla R3 and S3 levels
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Align Camarilla levels to lower timeframe
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if close_val > R3_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif close_val < S3_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price breaks below Camarilla S3 (reversal signal)
            if close_val < S3_aligned[i]:
                exit_signal = True
            # Exit: 1d trend changes to downtrend
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
            # Exit: price breaks above Camarilla R3 (reversal signal)
            if close_val > R3_aligned[i]:
                exit_signal = True
            # Exit: 1d trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals