#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Uses ATR(24) trailing stop for risk management. Discrete sizing 0.25 to balance return and fee drag.
# Target: 75-200 total trades over 4 years (19-50/year). Camarilla pivots provide structure in ranging markets,
# breakouts capture trends. Works in bull via long breakouts, in bear via short signals.
# Proven pattern from top performers: Camarilla pivot + HTF trend + volume confirmation + ATR stop.

name = "4h_Camarilla_R3_S3_12hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels from prior completed 12h bar
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    prior_high_12h = np.roll(df_12h['high'].values, 1)
    prior_high_12h[0] = np.nan
    prior_low_12h = np.roll(df_12h['low'].values, 1)
    prior_low_12h[0] = np.nan
    prior_close_12h = np.roll(df_12h['close'].values, 1)
    prior_close_12h[0] = np.nan
    
    # Calculate Camarilla levels: R3, S3
    camarilla_range = prior_high_12h - prior_low_12h
    camarilla_r3 = prior_close_12h + 1.1 * camarilla_range / 4
    camarilla_s3 = prior_close_12h - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
        # Get current values
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(camarilla_r3_val) or np.isnan(camarilla_s3_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above Camarilla R3 with volume spike and above 12h EMA50
        long_entry = (close[i] > camarilla_r3_val) and (close[i] > ema_trend) and vol_spike
        # Short: break below Camarilla S3 with volume spike and below 12h EMA50
        short_entry = (close[i] < camarilla_s3_val) and (close[i] < ema_trend) and vol_spike
        
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
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals