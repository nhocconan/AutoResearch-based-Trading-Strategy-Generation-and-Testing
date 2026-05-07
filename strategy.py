#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Combines Camarilla pivot levels from daily data with 1-day EMA trend filter and volume confirmation.
# Goes long when price breaks above R3 with bullish daily trend and volume spike.
# Goes short when price breaks below S3 with bearish daily trend and volume spike.
# Uses tight stop via EMA reversal and volume drop. Designed for 6h timeframe to capture multi-day moves
# while avoiding noise. Works in both bull and bear markets by following the daily trend.
# Target: 15-30 trades/year to stay within optimal frequency range.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    # We need previous day's H, L, C
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to get previous day's values (avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Set first day's previous values to NaN (no prior day)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels based on previous day
    camarilla_r3 = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.2500)
    camarilla_s3 = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.2500)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume spike on 6h timeframe (24-period average = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)  # Require strong volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(ema_34_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + bullish daily trend (price > EMA34) + volume spike
            if close[i] > camarilla_r3_6h[i] and close[i] > ema_34_1d_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + bearish daily trend (price < EMA34) + volume spike
            elif close[i] < camarilla_s3_6h[i] and close[i] < ema_34_1d_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below EMA34 (trend reversal) OR volume drops significantly
            if close[i] < ema_34_1d_6h[i] or volume[i] < (0.5 * vol_ma_24[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above EMA34 (trend reversal) OR volume drops significantly
            if close[i] > ema_34_1d_6h[i] or volume[i] < (0.5 * vol_ma_24[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals