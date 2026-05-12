#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 12h timeframe, enter long when price closes above Camarilla R3 level with price > 1d EMA34 and volume spike (1.5x 20-period MA).
# Enter short when price closes below Camarilla S3 level with price < 1d EMA34 and volume spike.
# Exit when price crosses 1d EMA34 (trend reversal).
# Uses daily timeframe for trend filter and volume confirmation to avoid false breakouts in sideways markets.
# Camarilla R3/S3 are strong resistance/support levels; breakouts with volume and trend alignment yield high-probability trades.
# Targets 15-30 trades/year for low fee drag and works in both bull and bear markets by following the daily trend.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot, trend filter, and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R3 and S3 are key breakout levels)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla R3 and S3 levels
    r3 = daily_pivot + daily_range * 1.1 / 4.0
    s3 = daily_pivot - daily_range * 1.1 / 4.0
    
    # Daily EMA34 for trend filter
    daily_ema34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period moving average on daily volume
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(daily_ema34_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        daily_trend = daily_ema34_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # LONG: Price closes above R3 with price > daily EMA34 and volume > 1.5x MA
            if close[i] > r3_val and close[i] > daily_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with price < daily EMA34 and volume > 1.5x MA
            elif close[i] < s3_val and close[i] < daily_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below daily EMA34 (trend reversal)
            if close[i] < daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above daily EMA34 (trend reversal)
            if close[i] > daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals