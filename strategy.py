#!/usr/bin/env python3
# 12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Breakouts above daily Camarilla R3 in uptrend (price > EMA34) and breakdowns below S3 in downtrend (price < EMA34), with volume confirmation.
# Uses 12h primary timeframe with 1d trend filter. Designed for low trade frequency to avoid fee drag in both bull and bear markets.

name = "12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
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
    
    # Get 1d data for Camarilla levels (based on previous day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's range)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    camarilla_r3 = np.full_like(prev_close, np.nan)
    camarilla_s3 = np.full_like(prev_close, np.nan)
    
    camarilla_r3[valid_idx] = prev_close[valid_idx] + 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 4
    camarilla_s3[valid_idx] = prev_close[valid_idx] - 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation in uptrend (price > EMA34)
            if camarilla_r3_aligned[i] > 0 and not np.isnan(camarilla_r3_aligned[i]) and \
               high[i] > camarilla_r3_aligned[i] and volume_confirmed[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation in downtrend (price < EMA34)
            elif camarilla_s3_aligned[i] > 0 and not np.isnan(camarilla_s3_aligned[i]) and \
                 low[i] < camarilla_s3_aligned[i] and volume_confirmed[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R3 or trend weakens (price < EMA34)
            if camarilla_r3_aligned[i] > 0 and not np.isnan(camarilla_r3_aligned[i]) and \
               low[i] < camarilla_r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S3 or trend weakens (price > EMA34)
            if camarilla_s3_aligned[i] > 0 and not np.isnan(camarilla_s3_aligned[i]) and \
               high[i] > camarilla_s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals