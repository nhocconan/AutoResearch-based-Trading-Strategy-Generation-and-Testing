#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Uses Camarilla R3/S3 levels from daily candles for structure, 1d EMA34 for trend filter,
# and volume spike for confirmation. Designed to capture strong intraday moves while
# avoiding choppy markets via trend alignment. Works in both bull and bear by
# following the 1d EMA direction.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from previous 1d bar (R3, S3)
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    close_1d_prev = df_1d['close'].shift(1).values  # previous day close
    high_1d_prev = df_1d['high'].shift(1).values   # previous day high
    low_1d_prev = df_1d['low'].shift(1).values     # previous day low
    camarilla_range = (high_1d_prev - low_1d_prev)
    camarilla_range = np.where(camarilla_range == 0, 1e-10, camarilla_range)
    camarilla_R3 = close_1d_prev + 1.1 * camarilla_range / 2
    camarilla_S3 = close_1d_prev - 1.1 * camarilla_range / 2
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Calculate volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        ema_trend = ema_34_aligned[i]
        camarilla_R3 = camarilla_R3_aligned[i]
        camarilla_S3 = camarilla_S3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(camarilla_R3) or np.isnan(camarilla_S3):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Entry conditions: breakout of Camarilla R3/S3 with volume spike and trend alignment
        long_entry = (close[i] > camarilla_R3) and vol_spike and (close[i] > ema_trend)
        short_entry = (close[i] < camarilla_S3) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on close below Camarilla S3 (mean reversion) or trend change
            if close[i] < camarilla_S3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on close above Camarilla R3 (mean reversion) or trend change
            if close[i] > camarilla_R3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals