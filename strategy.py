#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h ADX trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with 4h ADX > 20 (trending) and volume > 1.5x average.
# Short when price breaks below Camarilla S3 with 4h ADX > 20 and volume > 1.5x average.
# Uses ATR(14) trailing stop (2.5x) for risk control. Discrete sizing 0.20.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# ADX filter ensures we only trade strong trends, reducing whipsaw in ranging markets.
# Camarilla R3/S3 levels provide institutional breakout/breakdown points.
# Session filter (08-20 UTC) reduces noise trades during low-volume periods.

name = "1h_Camarilla_R3_S3_Breakout_4hADX_VolumeSpike_ATRStop_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for Camarilla pivot calculation and ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels from previous 4h bar
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    camarilla_r3 = np.full(len(close_4h), np.nan)
    camarilla_s3 = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Use previous 4h bar's data to calculate current levels
        prev_high = high_4h[i-1]
        prev_low = low_4h[i-1]
        prev_close = close_4h[i-1]
        
        camarilla_r3[i] = prev_close + ((prev_high - prev_low) * 1.1 / 4)
        camarilla_s3[i] = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 4h ADX(14) for trend filter
    # TR already calculated above for 4h
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    
    # +DM and -DM
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_4h = wilders_smoothing(tr_4h, 14)
    plus_dm_4h = wilders_smoothing(plus_dm, 14)
    minus_dm_4h = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di_4h = np.where(atr_4h != 0, (plus_dm_4h / atr_4h) * 100, 0)
    minus_di_4h = np.where(atr_4h != 0, (minus_dm_4h / atr_4h) * 100, 0)
    
    # Calculate DX and ADX
    dx_4h = np.where((plus_di_4h + minus_di_4h) != 0, 
                     np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h) * 100, 0)
    adx_4h = wilders_smoothing(dx_4h, 14)
    
    # Align 4h ADX to 1h timeframe (wait for 4h bar to close)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Precompute session filter (08-20 UTC)
    # open_time is already datetime64[ms], so we can use .hour directly
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(adx_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            # Carry forward tracking values when flat
            if i > 0 and position == 0:
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND 4h ADX > 20 AND volume > 1.5x average
            if (close[i] > camarilla_r3_aligned[i] and 
                adx_4h_aligned[i] > 20 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below Camarilla S3 AND 4h ADX > 20 AND volume > 1.5x average
            elif (close[i] < camarilla_s3_aligned[i] and 
                  adx_4h_aligned[i] > 20 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.20
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.20
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals