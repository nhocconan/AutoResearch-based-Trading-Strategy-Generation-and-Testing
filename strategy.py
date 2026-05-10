#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot levels act as strong support/resistance levels. 
# Breakout above R1 (resistance) or below S1 (support) with daily trend alignment and volume confirmation
# captures institutional breakouts. Works in bull markets (breakouts continue) and bear markets 
# (breakdowns continue). Uses 1-day trend filter to avoid counter-trend trades.
# Target: 20-50 trades/year to minimize fee drag.

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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (using prior day's data)
    # Camarilla formulas: 
    # R4 = close + ((high-low)*1.1/2)
    # R3 = close + ((high-low)*1.1/4)
    # R2 = close + ((high-low)*1.1/6)
    # R1 = close + ((high-low)*1.1/12)
    # PP = (high+low+close)/3
    # S1 = close - ((high-low)*1.1/12)
    # S2 = close - ((high-low)*1.1/6)
    # S3 = close - ((high-low)*1.1/4)
    # S4 = close - ((high-low)*1.1/2)
    prev_high = df_1d['high'].shift(1).values  # previous day's high
    prev_low = df_1d['low'].shift(1).values    # previous day's low
    prev_close = df_1d['close'].shift(1).values # previous day's close
    
    # Calculate Camarilla levels for previous day
    camarilla_R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (wait for daily candle to close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Daily trend filter: EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period volume MA on 4h chart (~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (needs 2 days), EMA50 (50), volume MA (20)
    start_idx = max(30, 50, 20)  # 30 to ensure we have 2 days of data
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 + uptrend + volume
            if close[i] > camarilla_R1_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + downtrend + volume
            elif close[i] < camarilla_S1_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend breaks
            if close[i] < camarilla_S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend breaks
            if close[i] > camarilla_R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals