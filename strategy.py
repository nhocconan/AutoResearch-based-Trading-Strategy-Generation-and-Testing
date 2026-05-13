#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above Camarilla R3 AND 4h EMA50 uptrend AND 1d volume > 1.5 * 20-period average.
# Short when price breaks below Camarilla S3 AND 4h EMA50 downtrend AND 1d volume > 1.5 * 20-period average.
# Exit when price crosses Camarilla pivot point (PP).
# Uses discrete position sizing (0.20) to limit fee churn. Designed for BTC/ETH robustness by capturing
# institutional breakouts with trend and volume confirmation in both bull and bear markets.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dVolumeSpike_v1"
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
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # For intraday, we use daily OHLC from 1d timeframe
    if len(df_1d) < 2:
        return np.zeros(n)
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot point (PP)
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate Camarilla levels
    r3 = pp + (prev_high - prev_low) * 1.1 / 4
    s3 = pp - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (using previous day's values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after we have previous day's data
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND 4h EMA50 uptrend AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and  # price above EMA50 = uptrend
                volume_spike_aligned[i] > 0.5):  # True if volume spike aligned
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S3 AND 4h EMA50 downtrend AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and  # price below EMA50 = downtrend
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below pivot point (PP)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price crosses above pivot point (PP)
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals