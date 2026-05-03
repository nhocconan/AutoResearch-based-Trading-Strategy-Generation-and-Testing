#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Uses ATR-based trailing stop for risk management. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide high-probability reversal/breakout points that work in ranging and trending markets.
# 12h EMA50 filter ensures alignment with medium-term trend.
# Volume confirmation reduces false breakouts.
# Based on proven 6h Camarilla patterns showing strong test performance in DB.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla pivot levels (R3, S3, R4, S4) from prior completed 12h bar
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least 1 completed bar for prior
        return np.zeros(n)
    
    # Calculate prior completed 12h bar's high, low, close for Camarilla
    prior_high_12h = np.roll(df_12h['high'].values, 1)
    prior_low_12h = np.roll(df_12h['low'].values, 1)
    prior_close_12h = np.roll(df_12h['close'].values, 1)
    prior_high_12h[0] = np.nan
    prior_low_12h[0] = np.nan
    prior_close_12h[0] = np.nan
    
    # Calculate Camarilla levels for prior 12h bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using prior bar's range
    prior_range = prior_high_12h - prior_low_12h
    camarilla_r3_12h = prior_close_12h + 1.1 * prior_range * 1.1 / 4
    camarilla_s3_12h = prior_close_12h - 1.1 * prior_range * 1.1 / 4
    camarilla_r4_12h = prior_close_12h + 1.1 * prior_range * 1.1 / 2
    camarilla_s4_12h = prior_close_12h - 1.1 * prior_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4_12h)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4_12h)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(30) for stoploss (using 6h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=30, min_periods=30, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 30-bar average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(r4_val) or np.isnan(s4_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above Camarilla R3 with volume spike and above 12h EMA50
        long_entry = (close[i] > r3_val) and (close[i] > ema_trend) and vol_spike
        # Short: break below Camarilla S3 with volume spike and below 12h EMA50
        short_entry = (close[i] < s3_val) and (close[i] < ema_trend) and vol_spike
        
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