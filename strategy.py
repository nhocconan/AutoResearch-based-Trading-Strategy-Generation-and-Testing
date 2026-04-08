#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_reversal
# Hypothesis: Camarilla pivot levels (from 1d) provide strong support/resistance. Price reverses from S3/R3 with volume confirmation.
# Long when price crosses above S3 (close > S3) with volume > 1.5x average and RSI(14) < 40.
# Short when price crosses below R3 (close < R3) with volume > 1.5x average and RSI(14) > 60.
# Exit when price reaches S1/R1 or opposite pivot level.
# Designed to work in ranging and trending markets by fading extremes with confirmation.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_reversal"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate RSI(14) on daily close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    # Prepend first value to match length
    rsi_1d = np.concatenate([np.full(1, 50.0), rsi_1d])
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We'll use previous day's data to avoid look-ahead
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate pivot components from previous day
    prev_open = df_1d['open'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    # S1 = C - ((H-L) * 1.1/6)
    # S2 = C - ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/2)
    # R2 = C + ((H-L) * 1.1/4)
    # R1 = C + ((H-L) * 1.1/6)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    
    # Align to 12h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S1 or R3 (take profit or stop)
            if close[i] >= camarilla_r1_aligned[i] or close[i] <= camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 or S3
            if close[i] <= camarilla_r1_aligned[i] or close[i] >= camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Mean reversion entries at extreme levels
            # Long: price crosses above S3 with oversold RSI
            if (close[i] > camarilla_s3_aligned[i]) and (rsi_1d_aligned[i] < 40) and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short: price crosses below R3 with overbought RSI
            elif (close[i] < camarilla_r3_aligned[i]) and (rsi_1d_aligned[i] > 60) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals