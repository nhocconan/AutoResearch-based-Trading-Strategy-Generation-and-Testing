#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike (1.8x)
# Long when price breaks above 1h Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume > 1.8x 20-period average
# Short when price breaks below 1h Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume > 1.8x 20-period average
# Exit when price crosses 1h Camarilla pivot point OR 4h EMA34 filter reverses
# Uses Camarilla pivots for structure + volume confirmation to reduce false breakouts
# 4h EMA34 provides strong trend filter for BTC/ETH in both bull and bear markets
# Session filter (08-20 UTC) reduces noise trades
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag on 1h timeframe
# Timeframe: 1h (primary), HTF: 4h

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeSpike_1.8x"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1h data for Camarilla calculation (based on previous day's OHLC)
    if len(close) < 2:
        return np.zeros(n)
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current values as fallback
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate Camarilla levels for 1h timeframe
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, Pivot = (H+L+C)/3
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4.0
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4.0
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Get 4h data ONCE before loop for EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(34)
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation on 1h (threshold: 1.8x for balanced frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot OR price < EMA34 (trend weakening)
            if close[i] < camarilla_pivot_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot OR price > EMA34 (trend weakening)
            if close[i] > camarilla_pivot_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals