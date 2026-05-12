#!/usr/bin/env python3
# 4H_CAMARILLA_R3_S3_BREAKOUT_12HTREND_VOLUME
# Hypothesis: Camarilla pivot levels (R3/S3) on 1d combined with 12h EMA trend and volume confirmation.
# Long when price closes above R3 with volume > 1.5x average and price above 12h EMA.
# Short when price closes below S3 with volume > 1.5x average and price below 12h EMA.
# Exit when price returns to opposite pivot level (R1 for longs, S1 for shorts) or trend invalidates.
# Designed to capture breakouts in trending markets with volume confirmation.
# Targets 25-40 trades/year to minimize fee drain with high-probability setups.

name = "4H_CAMARILLA_R3_S3_BREAKOUT_12HTREND_VOLUME"
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
    
    # Calculate Camarilla levels from previous 1d candle
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each 4h bar (based on previous day)
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    pclose_12h = df_12h['close'].values
    ema12h = pd.Series(pclose_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema12h_aligned = align_htf_to_ltf(prices, df_12h, ema12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema12h_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close above R3 with volume confirmation and uptrend
            if close[i] > R3[i] and vol_confirm[i] and close[i] > ema12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 with volume confirmation and downtrend
            elif close[i] < S3[i] and vol_confirm[i] and close[i] < ema12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to R1 or trend breaks
            if close[i] < R1[i] or close[i] < ema12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to S1 or trend breaks
            if close[i] > S1[i] or close[i] > ema12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals