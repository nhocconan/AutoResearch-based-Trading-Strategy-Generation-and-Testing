#!/usr/bin/env python3
name = "1d_1w_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d close for weekly aggregation (used in weekly high/low)
    close_1d = df_1d['close'].values
    
    # 1w high, low, close for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla levels from previous week
    # R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan  # First week has no previous
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    camarilla_r3_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 2
    camarilla_s3_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 2
    
    # Align weekly Camarilla levels to daily timeframe
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # Weekly trend: 50-period EMA on weekly close
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume spike: > 2.5x 20-day average
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume > 2.5 * vol_ma_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for weekly EMA50 and daily volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_1w_aligned[i]) or np.isnan(camarilla_s3_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly R3 with volume spike and price above weekly EMA50
            if (close[i] > camarilla_r3_1w_aligned[i] and vol_spike_1d[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S3 with volume spike and price below weekly EMA50
            elif (close[i] < camarilla_s3_1w_aligned[i] and vol_spike_1d[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below weekly S3 or price below weekly EMA50
            if close[i] < camarilla_s3_1w_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above weekly R3 or price above weekly EMA50
            if close[i] > camarilla_r3_1w_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Camarilla levels (R3/S3) act as strong support/resistance levels.
# Breakouts with volume confirmation and trend filter (price vs weekly EMA50) capture
# institutional breakout attempts. Works in both bull and bear markets by capturing
# strong directional moves after consolidation. Weekly timeframe reduces noise and
# false breakouts, while daily volume spike ensures genuine institutional interest.
# Position size 0.25 limits risk during volatile periods. Target ~15-25 trades/year
# to minimize fee drag while maintaining edge in BTC/ETH markets.