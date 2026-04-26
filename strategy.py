#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R3/S3 breakout with weekly trend filter (price vs 1w EMA34) and volume spike confirmation.
Long when price breaks above R3 with volume > 1.5x 20-day avg AND close > 1w EMA34.
Short when price breaks below S3 with volume > 1.5x 20-day avg AND close < 1w EMA34.
Uses discrete sizing (0.25) to minimize fee drag. Target: 30-100 trades over 4 years.
Works in bull/bear via weekly EMA trend filter and Camarilla levels as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate typical price for Camarilla levels
    typical_price = (high + low + close) / 3
    
    # Previous day's OHLC (needed for Camarilla calculation)
    # Since we're on 1d timeframe, we need to use previous day's values
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels calculation
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    rang = prev_high - prev_low
    r3 = prev_close + rang * 1.1 / 4
    s3 = prev_close - rang * 1.1 / 4
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period for volume MA and 1 for previous day)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R3 + volume spike + price > 1w EMA34
        long_condition = (close[i] > r3[i]) and volume_spike[i] and (close[i] > ema_34_1w_aligned[i])
        
        # Short logic: price breaks below S3 + volume spike + price < 1w EMA34
        short_condition = (close[i] < s3[i]) and volume_spike[i] and (close[i] < ema_34_1w_aligned[i])
        
        # Exit logic: reverse signal or volume dry-up
        exit_long = position == 1 and (close[i] < s3[i] or not volume_spike[i])
        exit_short = position == -1 and (close[i] > r3[i] or not volume_spike[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif exit_long:
            signals[i] = 0.0
            position = 0
        elif exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0