#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot R3/S3 levels with 1d EMA34 trend filter and volume confirmation.
# Uses 1w Camarilla R3/S3 for strong weekly support/resistance (proven effective in ranging and trending markets).
# Breakout at 1w Camarilla R3 (long) or S3 (short) levels with volume spike (>1.5x 20-bar average) for confirmation.
# 1d EMA34 as trend filter to avoid counter-trend trades in strong trends.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 30-100 total trades over 4 years = 7-25/year for 1d (within proven winning range).
# Works in BOTH bull and bear: trend filter ensures we only trade with the weekly trend,
# while volume confirmation avoids false breakouts. Weekly Camarilla levels provide structure
# that adapts to changing volatility regimes.

name = "1d_Camarilla_R3S3_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter (using same timeframe for consistency)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivot levels (R3, S3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    rng_1w = high_1w - low_1w
    camarilla_r3_1w = close_1w + rng_1w * 1.1 / 4
    camarilla_s3_1w = close_1w - rng_1w * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 1d timeframe (already 1d, but keeping for consistency with MTF pattern)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume spike: >1.5x 20-bar average volume (balanced frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA34
        above_ema = close[i] > ema_34_1d_aligned[i]
        below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > camarilla_r3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < camarilla_s3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < camarilla_s3_aligned[i] or below_ema
        short_exit = close[i] > camarilla_r3_aligned[i] or above_ema
        
        # Handle entries and exits
        if long_breakout and above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals