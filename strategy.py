#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Camarilla pivot levels (R3/S3) with volume confirmation and 1w EMA34 trend filter.
# Enter long when price touches or breaks above weekly S3 level with volume spike and above weekly EMA34.
# Enter short when price touches or breaks below weekly R3 level with volume spike and below weekly EMA34.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 total trades over 4 years.
# Camarilla levels provide structure, volume confirms breakout strength, EMA34 filters trend direction.
# Works in bull (breakouts with trend) and bear (failed breaks reverse) markets.

name = "1d_Camarilla_R3S3_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and EMA34
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R3, S3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    camarilla_r3 = np.full(n_1w, np.nan)
    camarilla_s3 = np.full(n_1w, np.nan)
    
    for i in range(n_1w):
        # Camarilla formula: range = high - low
        # R3 = high + 1.1 * (high - low) / 2
        # S3 = low - 1.1 * (high - low) / 2
        rng = high_1w[i] - low_1w[i]
        camarilla_r3[i] = high_1w[i] + 1.1 * rng / 2
        camarilla_s3[i] = low_1w[i] - 1.1 * rng / 2
    
    # Forward fill Camarilla levels
    camarilla_r3 = pd.Series(camarilla_r3).ffill().values
    camarilla_s3 = pd.Series(camarilla_s3).ffill().values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly data to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily volume spike: >2.0x 20-bar average volume (~1 month equivalent)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly EMA34
        above_ema = close[i] > ema_34_1w_aligned[i]
        below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Camarilla level touch/break conditions with volume confirmation
        long_breakout = close[i] >= camarilla_s3_aligned[i] and volume_spike[i]
        short_breakout = close[i] <= camarilla_r3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < camarilla_r3_aligned[i] or below_ema
        short_exit = close[i] > camarilla_s3_aligned[i] or above_ema
        
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