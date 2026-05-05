#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when: price > R3 AND 4h EMA50 rising AND volume > 1.5x 20-period MA
# Short when: price < S3 AND 4h EMA50 falling AND volume > 1.5x 20-period MA
# Exit when: price crosses middle (PP) OR volume drops below average
# Uses Camarilla for intraday structure, 4h EMA for trend, volume for conviction
# Timeframe: 1h, HTF: 4h. Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
# Session filter: 08-20 UTC to reduce noise trades.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels on 1h
    if len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
        # Use previous bar's high, low, close for today's levels
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        camarilla_pp = (prev_high + prev_low + prev_close) / 3
        camarilla_range = prev_high - prev_low
        camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4
        camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4
        camarilla_r4 = camarilla_pp + camarilla_range * 1.1 / 2
        camarilla_s4 = camarilla_pp - camarilla_range * 1.1 / 2
    else:
        camarilla_pp = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
        camarilla_r4 = np.full(n, np.nan)
        camarilla_s4 = np.full(n, np.nan)
    
    # Get 4h data ONCE before loop for EMA50 calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h
    close_4h = df_4h['close'].values
    if len(close_4h) >= 50:
        ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        # EMA rising/falling
        ema_rising = np.zeros(len(ema_50), dtype=bool)
        ema_falling = np.zeros(len(ema_50), dtype=bool)
        for i in range(1, len(ema_50)):
            if not np.isnan(ema_50[i]) and not np.isnan(ema_50[i-1]):
                ema_rising[i] = ema_50[i] > ema_50[i-1]
                ema_falling[i] = ema_50[i] < ema_50[i-1]
    else:
        ema_50 = np.full(len(close_4h), np.nan)
        ema_rising = np.zeros(len(close_4h), dtype=bool)
        ema_falling = np.zeros(len(close_4h), dtype=bool)
    
    # Align 4h EMA50 and trend to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    ema_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_falling.astype(float))
    
    # Volume confirmation on 1h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20, adjust=False).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pp[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 AND 4h EMA50 rising AND volume filter AND session
            if (close[i] > camarilla_r3[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: break below S3 AND 4h EMA50 falling AND volume filter AND session
            elif (close[i] < camarilla_s3[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below PP OR volume drops
            if (close[i] < camarilla_pp[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above PP OR volume drops
            if (close[i] > camarilla_pp[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals