#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level in bull trend (close > 4h EMA50) with volume > 1.8x 20-period MA.
# Short when price breaks below Camarilla S3 level in bear trend (close < 4h EMA50) with volume spike.
# Uses 1h timeframe for entry timing, 4h for trend direction and Camarilla levels.
# Session filter (08-20 UTC) to reduce noise trades. Position size 0.20.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

name = "1h_Camarilla_R3S3_4hEMA50_Volume_Session"
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
    
    # Get 4h data for EMA50 trend filter and Camarilla levels (based on prior 4h bar)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels for each 4h bar (based on prior 4h bar's range)
    prev_4h_high = df_4h['high'].values
    prev_4h_low = df_4h['low'].values
    prev_4h_close = df_4h['close'].values
    
    camarilla_r3 = prev_4h_close + (prev_4h_high - prev_4h_low) * 1.1 / 4
    camarilla_s3 = prev_4h_close - (prev_4h_high - prev_4h_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (no additional delay as based on completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume regime: current 1h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions
        breakout_r3 = close_val > r3_level
        breakout_s3 = close_val < s3_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_r3 and vol_spike:
                signals[i] = 0.20
                position = 1
            elif is_bear_trend and breakout_s3 and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 level OR trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R3 level OR trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals