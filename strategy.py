#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume spike: volume > 1.5 * EMA20 volume
    volume_1d = df_1d['volume'].values
    vol_ema20_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        vol_ema20_1d[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            vol_ema20_1d[i] = (volume_1d[i] * 2 + vol_ema20_1d[i-1] * 18) / 20
    
    vol_ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema20_1d)
    volume_spike = volume > vol_ema20_1d_aligned * 1.5
    
    # 4h Camarilla levels (based on previous 1d OHLC)
    # Camarilla levels for intraday trading
    # R4 = Close + (High - Low) * 1.1 / 2
    # R3 = Close + (High - Low) * 1.1 / 4
    # R2 = Close + (High - Low) * 1.1 / 6
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    # S2 = Close - (High - Low) * 1.1 / 6
    # S3 = Close - (High - Low) * 1.1 / 4
    # S4 = Close - (High - Low) * 1.1 / 2
    
    # We need previous day's OHLC for current 4h bar
    # Since we're on 4h timeframe, we'll use daily OHLC from 1d data
    # But we need to align it properly to 4h bars
    
    # Get daily OHLC
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    # We'll shift by 1 to use previous day's data
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_open_1d = np.roll(open_1d, 1)
    
    # First value will be invalid due to roll, set to nan
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_open_1d[0] = np.nan
    
    # Calculate Camarilla levels
    rang = prev_high_1d - prev_low_1d
    R1 = prev_close_1d + rang * 1.1 / 12
    S1 = prev_close_1d - rang * 1.1 / 12
    
    # Align to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(33, 0)  # Need EMA34 and Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        uptrend = close[i] > ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Uptrend + price breaks above R1 + volume spike
            if uptrend and close[i] > R1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below S1 + volume spike
            elif not uptrend and close[i] < S1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below S1
            if not uptrend or close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above R1
            if uptrend or close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals