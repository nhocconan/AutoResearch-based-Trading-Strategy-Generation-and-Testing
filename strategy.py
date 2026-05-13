#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA200 trend filter and 1d volume regime filter.
# Long when price breaks above R3, close > 4h EMA200, and 1d volume > 1.5x 20-day average.
# Short when price breaks below S3, close < 4h EMA200, and 1d volume > 1.5x 20-day average.
# Uses discrete sizing 0.20 to target 60-150 total trades over 4 years on 1h timeframe.
# Camarilla R3/S3 provide strong intraday breakout levels; 4h EMA200 filters for higher-timeframe trend;
# 1d volume regime ensures trades occur only during elevated participation, reducing false breakouts in low-volume environments.
# Works in bull markets via trend-aligned breakouts and in bear markets via mean-reversion at extreme levels with volume confirmation.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA200_1dVolumeRegime"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (R3, S3) using previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla for each 1d bar: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 = close + (high - low) * 1.1 / 4
    # Camarilla S3 = close - (high - low) * 1.1 / 4
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 4h EMA200 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    ema_200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 1-day volume average for regime filter (20-day)
    df_1d_vol = get_htf_data(prices, '1d')
    if len(df_1d_vol) < 2:
        return np.zeros(n)
    vol_20d_avg = pd.Series(df_1d_vol['volume']).rolling(window=20, min_periods=20).mean().shift(1).values
    vol_20d_avg_aligned = align_htf_to_ltf(prices, df_1d_vol, vol_20d_avg)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for volume average
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_20d_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3, close > 4h EMA200, 1d volume > 1.5x 20-day average
            if (high[i] > R3_aligned[i] and 
                close[i] > ema_200_4h_aligned[i] and 
                vol_20d_avg_aligned[i] > 0 and  # avoid division by zero
                df_1d_vol['volume'].iloc[df_1d_vol.index.get_loc(df_1d_vol.index[-1]) if len(df_1d_vol) > 0 else 0] > 1.5 * vol_20d_avg_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3, close < 4h EMA200, 1d volume > 1.5x 20-day average
            elif (low[i] < S3_aligned[i] and 
                  close[i] < ema_200_4h_aligned[i] and 
                  vol_20d_avg_aligned[i] > 0 and
                  df_1d_vol['volume'].iloc[df_1d_vol.index.get_loc(df_1d_vol.index[-1]) if len(df_1d_vol) > 0 else 0] > 1.5 * vol_20d_avg_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR 4h EMA200 trend fails
            if (low[i] < S3_aligned[i] or 
                close[i] < ema_200_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR 4h EMA200 trend fails
            if (high[i] > R3_aligned[i] or 
                close[i] > ema_200_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals