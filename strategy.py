#!/usr/bin/env python3
"""
4h_4H_1D_Camarilla_R3_S3_Breakout_With_Volume_Signal
Hypothesis: Daily Camarilla R3 and S3 levels act as key support/resistance. 
Breakouts above R3 or below S3 with volume confirmation capture trend moves. 
Mean reversion occurs when price returns to R3/S3. Designed for 15-25 trades/year 
to work in both bull and bear markets by combining breakout and mean reversion 
logic with volume filtering and daily trend filter.
"""

name = "4h_4H_1D_Camarilla_R3_S3_Breakout_With_Volume_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels: R3, R4, S3, S4"""
    range_ = high - low
    close_val = close
    r3 = close_val + (range_ * 1.1 / 4)
    s3 = close_val - (range_ * 1.1 / 4)
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend
    df_daily = get_htf_data(prices, '1d')
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate daily Camarilla R3 and S3
    r3_daily, s3_daily = calculate_camarilla(daily_high, daily_low, daily_close)
    
    # Align daily Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_daily, r3_daily)
    s3_4h = align_htf_to_ltf(prices, df_daily, s3_daily)
    
    # Daily trend: EMA34
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Wait for EMA34 to be valid
        if position == 0:
            # BREAKOUT LONG: Price breaks above R3 with volume confirmation and uptrend
            if close[i] > r3_4h[i] and close[i-1] <= r3_4h[i-1] and volume_confirm[i] and close[i] > ema34_4h[i]:
                signals[i] = 0.25
                position = 1
            # BREAKOUT SHORT: Price breaks below S3 with volume confirmation and downtrend
            elif close[i] < s3_4h[i] and close[i-1] >= s3_4h[i-1] and volume_confirm[i] and close[i] < ema34_4h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to R3 (mean reversion) or breaks above R3 with weak volume
            if close[i] <= r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to S3 (mean reversion) or breaks below S3 with weak volume
            if close[i] >= s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals