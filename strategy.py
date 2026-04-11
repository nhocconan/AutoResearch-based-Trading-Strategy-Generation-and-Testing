#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v1
# Strategy: 4h Camarilla pivot level breakout with volume confirmation and 1-day ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (S3/S4 for shorts, R3/R4 for longs) act as strong support/resistance.
# Breakouts above R4 or below S4 with volume > 2x average and ADX > 25 indicate institutional breakout.
# Works in bull markets via long breakouts and bear markets via short breakdowns.
# Designed for low trade frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Camarilla pivot levels (based on previous day's range)
    # Calculate daily pivot points using previous day's OHLC
    prev_day_high = df_1d['high'].shift(1).values  # Previous day high
    prev_day_low = df_1d['low'].shift(1).values    # Previous day low
    prev_day_close = df_1d['close'].shift(1).values # Previous day close
    
    # Align daily data to 4h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # Calculate Camarilla levels
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    rng = prev_day_high_aligned - prev_day_low_aligned
    r4 = prev_day_close_aligned + (rng * 1.1 / 2)
    r3 = prev_day_close_aligned + (rng * 1.1 / 4)
    s3 = prev_day_close_aligned - (rng * 1.1 / 4)
    s4 = prev_day_close_aligned - (rng * 1.1 / 2)
    
    # 1-day ADX for trend filter (14-period)
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate DI values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r4[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(s4[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_avg_20[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Camarilla breakout signals
        breakout_up = close[i] > r4[i]      # Break above R4
        breakdown_down = close[i] < s4[i]   # Break below S4
        
        # Entry conditions
        # Long: Break above R4 AND volume confirmation AND strong trend
        if breakout_up and vol_confirm and strong_trend and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Break below S4 AND volume confirmation AND strong trend
        elif breakdown_down and vol_confirm and strong_trend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price returns to pivot area (between S3 and R3)
        elif position == 1 and close[i] < r3[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > s3[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals