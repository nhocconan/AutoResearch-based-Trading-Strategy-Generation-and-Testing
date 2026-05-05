#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1w EMA34 trend filter + volume confirmation
# Long when price breaks above Camarilla R3 AND close > 1w EMA34 AND volume > 2.0x 24-period average
# Short when price breaks below Camarilla S3 AND close < 1w EMA34 AND volume > 2.0x 24-period average
# Exit when price crosses 1w EMA34 (trend reversal) OR touches opposite Camarilla level (S3 for long, R3 for short)
# Uses 12h primary timeframe with 1w HTF for trend filter to capture sustained moves
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Camarilla provides precise intraday levels; 1w EMA34 filters for higher-timeframe trend; volume confirms participation
# Works in bull markets via breakouts and in bear markets via trend-filtered shorts

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivot calculation (use previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 as breakout levels
    prev_close = np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].values[:-1]])
    prev_high = np.concatenate([[df_1d['high'].iloc[0]], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[df_1d['low'].iloc[0]], df_1d['low'].values[:-1]])
    
    # Align 1d data to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Calculate Camarilla R3 and S3
    rang = prev_high_aligned - prev_low_aligned
    camarilla_r3 = prev_close_aligned + 1.1 * rang
    camarilla_s3 = prev_close_aligned - 1.1 * rang
    
    # Volume confirmation: volume > 2.0x 24-period average (2*12h = 1 day)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (2.0 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND close > 1w EMA34 AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND close < 1w EMA34 AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA34 (trend reversal) OR touches Camarilla S3 (support)
            if close[i] < ema_34_1w_aligned[i] or close[i] <= camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA34 (trend reversal) OR touches Camarilla R3 (resistance)
            if close[i] > ema_34_1w_aligned[i] or close[i] >= camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals