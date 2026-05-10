#!/usr/bin/env python3
# 1h_Camarilla_R3S3_Breakout_4hTrend_Volume
# Hypothesis: Camarilla pivot levels on 1h provide precise entry/exit points. Trend filter from 4h EMA50 ensures trades follow higher timeframe direction. Volume confirmation reduces false breakouts. Designed for 1h timeframe with ~20-40 trades/year to minimize fee drag while capturing momentum in both bull and bear markets.

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # Camarilla levels for 1h (using previous bar's OHLC)
    # Calculate for each bar using previous bar's data
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    
    for i in range(1, n):
        # Use previous bar's OHLC to calculate current bar's levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        camarilla_r3[i] = prev_close + range_val * 1.1 / 4
        camarilla_s3[i] = prev_close - range_val * 1.1 / 4
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above R3 with 4h uptrend and volume
            if (close[i] > camarilla_r3[i] and 
                trend_4h_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S3 with 4h downtrend and volume
            elif (close[i] < camarilla_s3[i] and 
                  trend_4h_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit when price returns to S3 level or trend fails
            if (close[i] < camarilla_s3[i] or 
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit when price returns to R3 level or trend fails
            if (close[i] > camarilla_r3[i] or 
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals