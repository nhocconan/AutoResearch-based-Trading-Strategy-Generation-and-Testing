#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Camarilla R3, S3 levels (based on previous day's range)
    # Calculate previous day's range for 12h bars
    # For each 12h bar, we need the previous day's high/low/close
    # We'll compute daily high/low/close first
    daily_high = pd.Series(df_1d['high']).values
    daily_low = pd.Series(df_1d['low']).values
    daily_close = pd.Series(df_1d['close']).values
    
    # Calculate Camarilla levels for each day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 12h volume spike (24-period average - 2 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]) or np.isnan(vol_ma_24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla R3 with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > camarilla_r3_12h[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 with volume and 1d downtrend
            elif close[i] < camarilla_s3_12h[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Camarilla S3 or volume drops
            if close[i] < camarilla_s3_12h[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Camarilla R3 or volume drops
            if close[i] > camarilla_r3_12h[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA trend filter and volume confirmation
# - Camarilla R3/S3 levels act as strong support/resistance (proven in DB)
# - Breakouts with volume confirm institutional participation
# - 1d EMA(34) ensures alignment with higher timeframe trend
# - Works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend)
# - Position size 0.25 targets 15-25 trades/year, avoiding fee drag
# - Exit at opposite Camarilla level provides logical target in ranging markets
# - Uses 1d timeframe for Camarilla calculation (proven effective in DB)