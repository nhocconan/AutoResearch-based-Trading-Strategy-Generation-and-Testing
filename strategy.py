#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND close > 4h EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below Camarilla S3 AND close < 4h EMA50 AND volume > 1.5x 20-bar avg
# Exit when price retreats to Camarilla pivot point (PP) or volume drops
# Target: 15-37 trades/year via tight breakout conditions + trend/volume filters
# Works in both bull and bear markets by trading breakouts with trend confirmation

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h close
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 4h data (using previous day's OHLC)
    # Camarilla levels based on previous 4h bar's range
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = np.roll(close_4h, 1)
    close_4h_prev[0] = np.nan  # First value has no previous
    
    # Calculate pivot and ranges
    pp_4h = (high_4h + low_4h + close_4h_prev) / 3
    range_4h = high_4h - low_4h
    
    # Camarilla levels
    r3_4h = pp_4h + range_4h * 1.1/4
    s3_4h = pp_4h - range_4h * 1.1/4
    r4_4h = pp_4h + range_4h * 1.1/2
    s4_4h = pp_4h - range_4h * 1.1/2
    
    # Align 4h indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(pp_4h_aligned[i]) or 
            np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_4h_aligned[i]
        price = close[i]
        pp = pp_4h_aligned[i]
        r3 = r3_4h_aligned[i]
        s3 = s3_4h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > EMA50 (uptrend) AND volume confirmation
            if price > r3 and price > ema_trend and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3 AND price < EMA50 (downtrend) AND volume confirmation
            elif price < s3 and price < ema_trend and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retreats to PP or volume drops
            if price < pp or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when price retreats to PP or volume drops
            if price > pp or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals