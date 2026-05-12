#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
# Hypothesis: Use Camarilla R3/S3 breakout with 12h EMA trend filter and volume spike (2x MA).
# Long when price breaks above R3 with price > 12h EMA and volume > 2x MA.
# Short when price breaks below S3 with price < 12h EMA and volume > 2x MA.
# Exit when price closes back inside R3/S3 range.
# Designed to capture strong breakouts with trend and volume confirmation.
# Targets 25-40 trades/year to minimize fee drag while maintaining edge.

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
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
    
    # Calculate Camarilla levels (based on previous day's range)
    # For intraday, we use previous 4h bar's high/low to calculate today's levels
    # But since we're on 4h timeframe, we calculate based on previous candle
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R3 and S3 levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    twelve_h_close = df_12h['close'].values
    twelve_h_ema = pd.Series(twelve_h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    twelve_h_ema_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(twelve_h_ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with price > 12h EMA and volume > 2x MA
            if close[i] > R3[i] and close[i] > twelve_h_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with price < 12h EMA and volume > 2x MA
            elif close[i] < S3[i] and close[i] < twelve_h_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes back below R3 (reversion to mean)
            if close[i] < R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back above S3 (reversion to mean)
            if close[i] > S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals