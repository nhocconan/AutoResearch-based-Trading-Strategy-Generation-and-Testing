#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND 1w close > 1w open (bullish week) AND volume > 1.5x 20-bar average.
# Short when price breaks below Camarilla S3 AND 1w close < 1w open (bearish week) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts.
# Volume spike threshold set to 1.5x to avoid choppy market noise while capturing momentum.
# Primary timeframe: 12h, HTF: 1w for weekly bias.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: 1 = bullish week (close > open), -1 = bearish week (close < open)
    weekly_trend_raw = np.where(df_1w['close'].values > df_1w['open'].values, 1,
                                np.where(df_1w['close'].values < df_1w['open'].values, -1, 0))
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_raw)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous completed day
    # Camarilla R3 = close + 1.1*(high - low)/2
    # Camarilla S3 = close - 1.1*(high - low)/2
    camarilla_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    
    # Shift by 1 to use only completed day's levels
    camarilla_r3_shifted = camarilla_r3.shift(1).values
    camarilla_s3_shifted = camarilla_s3.shift(1).values
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    
    # Volume confirmation: current 12h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above Camarilla R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below Camarilla S3
        
        # Weekly trend filter
        bullish_week = weekly_trend_aligned[i] == 1
        bearish_week = weekly_trend_aligned[i] == -1
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Camarilla R3 AND bullish week AND volume confirmation
            if (breakout_up and 
                bullish_week and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Camarilla S3 AND bearish week AND volume confirmation
            elif (breakout_down and 
                  bearish_week and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Camarilla S3 (stoploss) OR weekly trend turns bearish
            if (curr_low < camarilla_s3_aligned[i] or 
                weekly_trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla R3 (stoploss) OR weekly trend turns bullish
            if (curr_high > camarilla_r3_aligned[i] or 
                weekly_trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals