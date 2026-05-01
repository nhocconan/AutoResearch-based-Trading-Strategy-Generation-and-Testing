#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 1d data to identify key support/resistance (R3/S3 for breakouts, R4/S4 for strong continuation)
# EMA34 on 1d determines structural bias (long above EMA34, short below)
# Volume spike > 2.0x 20-period EMA confirms institutional participation in breakouts
# Designed for low trade frequency: ~15-30 trades/year per symbol with 0.25 sizing
# Camarilla R3/S3 breakouts in direction of 1d EMA34 trend have high follow-through in both bull and bear markets
# Works in ranging markets (fade at R3/S3) and trending markets (breakout continuation)

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Camarilla: Pivot = (HIGH + LOW + CLOSE) / 3
    # Range = HIGH - LOW
    # R3 = Pivot + Range * 1.1/2
    # S3 = Pivot - Range * 1.1/2
    # R4 = Pivot + Range * 1.1
    # S4 = Pivot - Range * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + (range_1d * 1.1 / 2)
    s3_1d = pivot_1d - (range_1d * 1.1 / 2)
    r4_1d = pivot_1d + (range_1d * 1.1)
    s4_1d = pivot_1d - (range_1d * 1.1)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d EMA34 (34 bars) + volume EMA20 + Camarilla (uses prior 1d)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_trend = close[i] > ema34_aligned[i]
        bearish_trend = close[i] < ema34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_trend:
                # Long: break above R3 with volume spike (continuation)
                # or break above R4 with volume spike (strong breakout)
                if ((close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1]) or
                    (close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1])) and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_trend:
                # Short: break below S3 with volume spike (continuation)
                # or break below S4 with volume spike (strong breakdown)
                if ((close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1]) or
                    (close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1])) and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA34
        
        elif position == 1:  # Long position
            # Exit: break below S3 (failure of bullish structure)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: break above R3 (failure of bearish structure)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals