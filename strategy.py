#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 4h volume confirmation (>2.0x 20-period average).
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 (bullish trend) AND volume > 2.0x MA20.
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 (bearish trend) AND volume > 2.0x MA20.
# Exit when price retests the broken Camarilla level (R3 for long, S3 for short) OR closes beyond the opposite Camarilla level (S3 for long, R3 for short).
# Uses 1d HTF for trend to reduce noise and overtrading. Volume confirmation (>2.0x) reduces false signals.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe.
# Camarilla levels provide precise intraday support/resistance; breakouts with volume and HTF trend filter capture strong moves.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_4hVolumeConfirm_v1"
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
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 1d (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low),
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels for current 4h bars)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r3_1d_aligned[i]) or
            np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(volume_confirm_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND close > 1d EMA34 (bullish trend) AND volume confirm
            if (high[i] > camarilla_r3_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 AND close < 1d EMA34 (bearish trend) AND volume confirm
            elif (low[i] < camarilla_s3_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests broken R3 level (now support) OR closes below S3 (strong reversal)
            if (low[i] <= camarilla_r3_1d_aligned[i] or 
                close[i] < camarilla_s3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests broken S3 level (now resistance) OR closes above R3 (strong reversal)
            if (high[i] >= camarilla_s3_1d_aligned[i] or 
                close[i] > camarilla_r3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals