#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout + 4h EMA34 trend filter + volume spike (>1.8x 20-bar volume MA) + session filter (08-20 UTC)
# Camarilla pivot levels provide institutional support/resistance; breakout of R3/S3 with volume confirms strong momentum.
# 4h EMA34 ensures alignment with higher-timeframe trend to avoid counter-trend trades. Session filter reduces noise.
# Discrete sizing (0.20) minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA(34) on 4h close
    ema_4h_34 = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_4h_34_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_34)
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_val = high_prev - low_prev
    R3 = pivot + (range_val * 1.1 / 4)
    S3 = pivot - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 34 for 4h EMA and 20 for volume MA (50 > 34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_34_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Camarilla breakout conditions
        breakout_up = curr_close > R3_aligned[i-1]  # Break above prior period R3
        breakout_down = curr_close < S3_aligned[i-1]  # Break below prior period S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up, price above 4h EMA34, volume spike
            if breakout_up and curr_close > ema_4h_34_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Camarilla breakout down, price below 4h EMA34, volume spike
            elif breakout_down and curr_close < ema_4h_34_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakdown or price below 4h EMA34
            if curr_close < S3_aligned[i] or curr_close < ema_4h_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout or price above 4h EMA34
            if curr_close > R3_aligned[i] or curr_close > ema_4h_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals