#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R reversal + 1w EMA trend filter + volume confirmation.
# Long when 1d Williams %R < -80 (oversold) and price crosses above -50 level, with 1w EMA up and volume > 1.5x average.
# Short when 1d Williams %R > -20 (overbought) and price crosses below -50 level, with 1w EMA down and volume > 1.5x average.
# Uses discrete position size 0.25. Williams %R provides mean reversion in extremes, 1w EMA ensures trend alignment,
# volume confirmation avoids false breakouts. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation (14-period)
    williams_r = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        highest_high = np.max(high_1d[i-14:i+1])
        lowest_low = np.min(low_1d[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Williams %R signal: 1 for bullish cross above -50, -1 for bearish cross below -50, 0 otherwise
    williams_signal = np.zeros_like(close_1d)
    for i in range(15, len(close_1d)):
        prev_r = williams_r[i-1]
        curr_r = williams_r[i]
        if prev_r <= -50 and curr_r > -50:
            williams_signal[i] = 1   # bullish crossover
        elif prev_r >= -50 and curr_r < -50:
            williams_signal[i] = -1  # bearish crossover
    
    # Get 1w data once before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA (34-period) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_dir = np.zeros_like(ema_34_1w)
    ema_34_1w_dir[1:] = np.where(ema_34_1w[1:] > ema_34_1w[:-1], 1, np.where(ema_34_1w[1:] < ema_34_1w[:-1], -1, 0))
    
    # Align 1d Williams %R signal and 1w EMA direction to 6h timeframe
    williams_signal_aligned = align_htf_to_ltf(prices, df_1d, williams_signal)
    ema_34_1w_dir_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_dir)
    
    # Volume moving average (20-period) on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_signal_aligned[i]) or 
            np.isnan(ema_34_1w_dir_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        williams_sig = williams_signal_aligned[i]
        ema_dir = ema_34_1w_dir_aligned[i]
        vol_ma_val = vol_ma_20[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_val
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R signal turns bearish or EMA trend turns down
            if williams_sig <= 0 or ema_dir <= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R signal turns bullish or EMA trend turns up
            if williams_sig >= 0 or ema_dir >= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: 1w EMA must be trending (non-zero)
            trend_filter = ema_dir != 0
            
            # LONG: Williams %R bullish cross, EMA up, volume confirmation
            if (williams_sig > 0) and (ema_dir > 0) and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R bearish cross, EMA down, volume confirmation
            elif (williams_sig < 0) and (ema_dir < 0) and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dWilliamsR_1wEMA34_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0