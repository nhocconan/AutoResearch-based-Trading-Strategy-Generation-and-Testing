# 4h_12h_1d_Camarilla_R3S3_Breakout_TrendVolume_v1
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend and volume confirmation
# - Uses Camarilla levels from 1d timeframe for stability
# - Requires 12h EMA trend filter to align with higher timeframe momentum
# - Volume confirmation (1.5x 4-period average) reduces false breakouts
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to R3/S3 or volume weakens
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Designed for BTC/ETH primary focus with CAMARILLA structure proven effective

#!/usr/bin/env python3
name = "4h_12h_1d_Camarilla_R3S3_Breakout_TrendVolume_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S1 = Close - (High - Low) * 1.1/6
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    range_hl = prev_high - prev_low
    
    # Camarilla R3 and S3 levels (key reversal points)
    r3 = prev_close + range_hl * 1.1 / 4
    s3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection: 4-period average (1 day of 4h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S3 with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            uptrend = ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]
            
            if close[i] > s3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R3 with volume and 12h downtrend
            elif close[i] < r3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S3 or volume drops
            if close[i] < s3_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R3 or volume drops
            if close[i] > r3_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend and volume confirmation
# - Camarilla R3/S3 act as key support/resistance levels from previous day
# - Breakout above S3 with volume in 12h uptrend = long opportunity
# - Breakdown below R3 with volume in 12h downtrend = short opportunity
# - Volume spike (1.5x average) confirms institutional participation
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to S3/R3 or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses 1d Camarilla levels (not intraday) for better stability
# - 12h trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Camarilla R3/S3 (1d) + trend (12h) + volume (4h)
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits