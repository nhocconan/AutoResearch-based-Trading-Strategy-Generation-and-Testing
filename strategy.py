#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation.
# Camarilla pivot levels (R1/S1, R2/S2, R3/S3, R4/S4) act as strong support/resistance.
# Strategy: Fade at R3/S3 (mean reversion) when price shows rejection signals.
# Continuation at R4/S4 (breakout) when price breaks with volume.
# 1d EMA34 provides trend filter: only take longs in uptrend, shorts in downtrend.
# Volume confirmation avoids false breakouts/breakdowns.
# Designed for 6h timeframe to balance trade frequency and signal quality.
# Targets 15-30 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot and EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla pivot levels
    # Using yesterday's high, low, close to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first value to NaN since we don't have previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    r3 = pivot + (range_val * 1.1 / 4)
    r4 = pivot + (range_val * 1.1 / 2)
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_1d_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_spike = vol > 1.3 * vol_ma
        
        if position == 0:
            # Fade at R3/S3: mean reversion when price rejects extreme levels
            # Long: price rejects S3 and closes back above it (bullish rejection)
            # Short: price rejects R3 and closes back below it (bearish rejection)
            if (price > s3_aligned[i] and 
                prices['low'].iloc[i] <= s3_aligned[i] and 
                price > ema_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            elif (price < r3_aligned[i] and 
                  prices['high'].iloc[i] >= r3_aligned[i] and 
                  price < ema_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            # Breakout at R4/S4: continuation when price breaks with volume
            elif (price > r4_aligned[i] and 
                  price > ema_val and 
                  vol_spike):
                signals[i] = 0.25
                position = 1
            elif (price < s4_aligned[i] and 
                  price < ema_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches S4 (strong support) or R4 (profit target)
                # or when trend changes
                if (price <= s4_aligned[i] or 
                    price >= r4_aligned[i] or 
                    price < ema_val):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price reaches R4 (strong resistance) or S4 (profit target)
                # or when trend changes
                if (price >= r4_aligned[i] or 
                    price <= s4_aligned[i] or 
                    price > ema_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0