#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) from daily candles combined with weekly trend filter (close > EMA20) and volume confirmation (volume > 1.5x 20-period average) provides high-probability breakout trades. Daily timeframe reduces trade frequency to avoid fee drag, while weekly trend ensures alignment with higher-timeframe momentum. Works in both bull and bear markets by capturing breakouts in trending regimes.
"""

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(20) on weekly close for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # We use previous day's data to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current bar (no previous)
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous day's data
        if position == 0:
            # LONG: Close breaks above R3, weekly uptrend, volume confirmation
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema20_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3, weekly downtrend, volume confirmation
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close drops below S3 (reversal) OR weekly trend turns down
            if (close[i] < camarilla_s3[i]) or (close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises above R3 (reversal) OR weekly trend turns up
            if (close[i] > camarilla_r3[i]) or (close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals