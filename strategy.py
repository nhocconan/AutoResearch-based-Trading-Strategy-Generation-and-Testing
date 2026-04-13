#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
    # Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe
    # Uses 1d for pivot levels and volume, 1w for trend direction
    # Works in both bull and bear: breakout continuation in trending markets, avoids false breakouts in ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d bar)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Camarilla levels (R3/S3 for fade, R4/S4 for breakout)
    camarilla_r3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_r4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_s4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        # Get the 1d bar index for current 6h bar (each 1d bar = 4 6h bars)
        idx_1d = i // 4
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions at Camarilla R4/S4 levels
        breakout_long = close[i] > camarilla_r4_aligned[i]  # Price above R4 -> long breakout
        breakout_short = close[i] < camarilla_s4_aligned[i]  # Price below S4 -> short breakout
        
        # Fade conditions at Camarilla R3/S3 levels (counter-trend)
        fade_long = close[i] < camarilla_s3_aligned[i]  # Price below S3 -> long (expect bounce up)
        fade_short = close[i] > camarilla_r3_aligned[i]  # Price above R3 -> short (expect bounce down)
        
        # Trend filter: only trade breakout in direction of 1w EMA20, fade against trend
        trend_filter_long = close[i] > ema20_1w_aligned[i]  # Uptrend
        trend_filter_short = close[i] < ema20_1w_aligned[i]  # Downtrend
        
        # Entry conditions: breakout with trend OR fade against trend (with volume)
        enter_long = (breakout_long and trend_filter_long and volume_confirmed) or \
                     (fade_long and not trend_filter_long and volume_confirmed)
        enter_short = (breakout_short and trend_filter_short and volume_confirmed) or \
                      (fade_short and not trend_filter_short and volume_confirmed)
        
        # Exit conditions: price returns to opposite Camarilla level
        exit_long = position == 1 and close[i] <= camarilla_s3_aligned[i]
        exit_short = position == -1 and close[i] >= camarilla_r3_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_camarilla_breakout_fade_volume_trend_v1"
timeframe = "6h"
leverage = 1.0