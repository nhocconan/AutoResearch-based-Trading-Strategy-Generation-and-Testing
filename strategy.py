# 4H CAMARILLA R3-S3 BREAKOUT WITH VOLUME FILTER
# Based on top-performing patterns: Camarilla pivot levels from daily data + volume spike + trend filter
# Strategy targets 20-40 trades/year (80-160 over 4 years) to avoid fee drag
# Works in both bull/bear markets: breaks key intraday resistance/support with institutional volume confirmation
# Uses 1d Camarilla levels (R3/S3) as breakout levels, volume > 1.5x average, and price > EMA50 for trend filter
# Entry: break above R3 with volume + uptrend OR break below S3 with volume + downtrend
# Exit: return to pivot point (P) or opposite Camarilla level

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels (HIGHER TIMEFRAME)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels for previous day (using OHLC from prior day)
    # Camarilla formulas: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    # Calculate for each day using previous day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll, handle it
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    rang = prev_high - prev_low
    
    # Calculate Camarilla levels
    R3 = prev_close + (rang * 1.1 / 4)
    S3 = prev_close - (rang * 1.1 / 4)
    PP = (prev_high + prev_low + prev_close) / 3
    
    # Align daily indicators to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 1.5)
    
    # Precompute session filter (08-20 UTC) for better liquidity
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(PP_aligned[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Entry conditions: Camarilla R3/S3 breakout with volume and trend
        long_breakout = close[i] > R3_aligned[i]
        short_breakout = close[i] < S3_aligned[i]
        
        long_entry = long_breakout and vol_filter[i] and uptrend
        short_entry = short_breakout and vol_filter[i] and downtrend
        
        # Exit conditions: return to pivot point (P) or opposite level
        long_exit = close[i] < PP_aligned[i]  # Return to pivot
        short_exit = close[i] > PP_aligned[i]  # Return to pivot
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_VolumeFilter_EMA50"
timeframe = "4h"
leverage = 1.0