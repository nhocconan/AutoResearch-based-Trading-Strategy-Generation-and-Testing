#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: 4H Camarilla R1/S1 breakout with daily trend filter and volume confirmation. Uses 1D trend to filter direction, reducing counter-trend trades. Volume surge confirms breakout strength. Designed for low trade frequency (19-50/year) to minimize fee drag in ranging markets.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4H OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 4H bar
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = (prev_high + prev_low + 2 * prev_close) / 3 + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = (prev_high + prev_low + 2 * prev_close) / 3 - (prev_high - prev_low) * 1.1 / 12
    
    # Daily trend: EMA(34) slope
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slope_34_1d = np.diff(ema_34_1d, prepend=ema_34_1d[0])
    ema_slope_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_34_1d)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i]) or
            np.isnan(ema_slope_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from daily EMA34 slope
        bullish_trend = ema_slope_34_1d_aligned[i] > 0
        bearish_trend = ema_slope_34_1d_aligned[i] < 0
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 in bullish trend with volume surge
            if close[i] > camarilla_r1[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in bearish trend with volume surge
            elif close[i] < camarilla_s1[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit: price crosses below previous day's close or trend change
                prev_daily_close = df_1d['close'].values
                prev_daily_close_aligned = align_htf_to_ltf(prices, df_1d, prev_daily_close)
                if close[i] < prev_daily_close_aligned[i] or ema_slope_34_1d_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: price crosses above previous day's close or trend change
                prev_daily_close = df_1d['close'].values
                prev_daily_close_aligned = align_htf_to_ltf(prices, df_1d, prev_daily_close)
                if close[i] > prev_daily_close_aligned[i] or ema_slope_34_1d_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals