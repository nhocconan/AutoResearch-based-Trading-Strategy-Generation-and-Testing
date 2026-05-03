#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Uses ATR(24) trailing stop for risk management. Discrete sizing 0.20 to balance return and fee drag.
# Target: 60-150 total trades over 4 years (15-37/year). Uses 4h/1d for signal direction, 1h only for entry timing.
# Session filter (08-20 UTC) reduces noise trades. Works in bull via long breakouts, in bear via short signals.

name = "1h_Camarilla_R1_S1_4hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate prior completed bar's high/low for Camarilla calculation
    prior_high = np.roll(high, 1)
    prior_high[0] = np.nan
    prior_low = np.roll(low, 1)
    prior_low[0] = np.nan
    
    # Calculate 1d Camarilla levels from prior completed 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prior_high_1d = np.roll(df_1d['high'].values, 1)
    prior_high_1d[0] = np.nan
    prior_low_1d = np.roll(df_1d['low'].values, 1)
    prior_low_1d[0] = np.nan
    prior_close_1d = np.roll(df_1d['close'].values, 1)
    prior_close_1d[0] = np.nan
    
    # Calculate Camarilla levels: R1, S1, R3, S3
    camarilla_range = prior_high_1d - prior_low_1d
    camarilla_r1 = prior_close_1d + 1.1 * camarilla_range / 4
    camarilla_s1 = prior_close_1d - 1.1 * camarilla_range / 4
    camarilla_r3 = prior_close_1d + 1.1 * camarilla_range / 2
    camarilla_s3 = prior_close_1d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(24) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, min_periods=24, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Get current values
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_trend = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(camarilla_r1_val) or np.isnan(camarilla_s1_val) or np.isnan(camarilla_r3_val) or np.isnan(camarilla_s3_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Entry conditions
        # Long: break above Camarilla R1 with volume spike and above 4h EMA50
        long_entry = (close[i] > camarilla_r1_val) and (close[i] > ema_trend) and vol_spike
        # Short: break below Camarilla S1 with volume spike and below 4h EMA50
        short_entry = (close[i] < camarilla_s1_val) and (close[i] < ema_trend) and vol_spike
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals