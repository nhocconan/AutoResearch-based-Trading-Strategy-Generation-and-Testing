#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h for signal direction (EMA50) and 1d for regime (above/below EMA200) to avoid counter-trend trades.
# Entry only on 1h break of Camarilla R3 (short) or S3 (long) with volume spike.
# Works in bull via breakout longs, in bear via breakdown shorts during rallies.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dEMA200_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels using previous day's OHLC
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L) * 1.1/2, S3 = C - (H-L) * 1.1/2
    # We need daily OHLC, so resample to 1d using get_htf_data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    camarilla_r3 = daily_close + (daily_range * 1.1 / 2.0)
    camarilla_s3 = daily_close - (daily_range * 1.1 / 2.0)
    
    # Align Camarilla levels to 1h timeframe (previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 4h EMA(50) for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(200) for regime filter (HTF) - only trade with trend
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50, 200)  # warmup for volume MA, 4h EMA50, 1d EMA200
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_ema_200_1d = ema_200_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above R3 AND above 4h EMA50 AND above 1d EMA200 (bullish regime)
                if (curr_close > curr_r3 and 
                    curr_close > curr_ema_50_4h and 
                    curr_close > curr_ema_200_1d):
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S3 AND below 4h EMA50 AND below 1d EMA200 (bearish regime)
                elif (curr_close < curr_s3 and 
                      curr_close < curr_ema_50_4h and 
                      curr_close < curr_ema_200_1d):
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below S3 (mean reversion) or closes below 4h EMA50
            if (curr_close < curr_s3 or curr_close < curr_ema_50_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price breaks above R3 (mean reversion) or closes above 4h EMA50
            if (curr_close > curr_r3 or curr_close > curr_ema_50_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals