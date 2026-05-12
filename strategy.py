#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Price breaking above Camarilla R3 (bullish) or below S3 (bearish) on 12h, with volume confirmation and 1d EMA trend filter, captures institutional breakouts in both bull and bear markets.
# Uses only 3 conditions to minimize trade frequency: Camarilla level breakout, volume spike (>1.5x 20-period average), and 1d EMA trend alignment.
# Targets 15-25 trades/year to avoid fee drag while maintaining high-probability setups.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate Camarilla levels for each bar using OHLC of that bar
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    hl_range = high - low
    camarilla_r3 = close + hl_range * 1.1 / 2.0
    camarilla_s3 = close - hl_range * 1.1 / 2.0
    
    # Volume filter: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > vol_ma * 1.5
    
    # 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema1d = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 with volume spike and uptrend
            if close[i] > camarilla_r3[i] and volume_spike[i] and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 with volume spike and downtrend
            elif close[i] < camarilla_s3[i] and volume_spike[i] and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 or trend breaks
            if close[i] < camarilla_s3[i] or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R3 or trend breaks
            if close[i] > camarilla_r3[i] or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals