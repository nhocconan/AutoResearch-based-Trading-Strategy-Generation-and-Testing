#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v2
Hypothesis: On 4h timeframe, use Camarilla pivot levels from 1d timeframe for mean reversion in ranging markets.
Enter long at S3 level when price closes below S3 and volume > 1.5x average, targeting reversion to S4.
Enter short at R3 level when price closes above R3 and volume > 1.5x average, targeting reversion to R4.
Only trade when market is ranging (ADX < 25 on 1d). Exit at opposite pivot level or when ADX > 25.
Uses tight entry conditions to target ~25-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v2"
timeframe = "4h"
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
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Camarilla pivot and ADX (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    camarilla_s3 = prev_close - (1.1 * range_val / 2)
    camarilla_s4 = prev_close - (1.1 * range_val)
    camarilla_r3 = prev_close + (1.1 * range_val / 2)
    camarilla_r4 = prev_close + (1.1 * range_val)
    
    # Calculate ADX on daily timeframe for regime filter
    # ADX components: +DI, -DI, DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (14-period)
    def smooth(val, period):
        result = np.full_like(val, np.nan)
        if len(val) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(val[:period])
            # Subsequent values use Wilder's smoothing
            for i in range(period, len(val)):
                result[i] = (result[i-1] * (period-1) + val[i]) / period
        return result
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / atr
    minus_di = 100 * smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth(dx, 14)
    
    # Align all daily indicators to 4h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: only trade when ranging (ADX < 25)
        ranging = adx_aligned[i] < 25
        
        if position == 1:  # Long position (expecting reversion to S4)
            # Exit conditions
            exit_long = False
            # Exit when price reaches S4 (target)
            if close[i] <= camarilla_s4_aligned[i]:
                exit_long = True
            # Exit when ADX > 25 (trending market)
            elif adx_aligned[i] >= 25:
                exit_long = True
            # Exit when price moves above R3 (failed reversion)
            elif close[i] >= camarilla_r3_aligned[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position (expecting reversion to R4)
            # Exit conditions
            exit_short = False
            # Exit when price reaches R4 (target)
            if close[i] >= camarilla_r4_aligned[i]:
                exit_short = True
            # Exit when ADX > 25 (trending market)
            elif adx_aligned[i] >= 25:
                exit_short = True
            # Exit when price moves below S3 (failed reversion)
            elif close[i] <= camarilla_s3_aligned[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price closes below S3, volume confirmation, ranging market
            long_entry = (close[i] < camarilla_s3_aligned[i]) and vol_confirm and ranging
            
            # Short entry: price closes above R3, volume confirmation, ranging market
            short_entry = (close[i] > camarilla_r3_aligned[i]) and vol_confirm and ranging
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals