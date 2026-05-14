#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 4h volume confirmation (>1.5x 20-period average).
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 (bearish trend) AND volume > 1.5x MA20.
# Exit when price returns to Camarilla Pivot Point (PP) OR trend reverses (close crosses 1d EMA34 opposite).
# Uses 1d HTF for trend to reduce noise and overtrading. Volume confirmation (>1.5x) reduces false signals.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe.
# Camarilla pivot levels provide intraday support/resistance; breakouts with volume and trend filter capture strong moves.

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
    # 4h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (1.5 * vol_ma_20)
    
    # --- Camarilla Pivot Levels (from previous day) ---
    # Need daily high, low, close from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    # Use previous day's OHLC to calculate today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: PP = (H+L+C)/3
    # R4 = PP + (H-L)*1.1/2, R3 = PP + (H-L)*1.1/4, R2 = PP + (H-L)*1.1/6, R1 = PP + (H-L)*1.1/12
    # S1 = PP - (H-L)*1.1/12, S2 = PP - (H-L)*1.1/6, S3 = PP - (H-L)*1.1/4, S4 = PP - (H-L)*1.1/2
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    r3 = pivot + (rang * 1.1 / 4.0)
    s3 = pivot - (rang * 1.1 / 4.0)
    pp = pivot  # Pivot Point for exit
    
    # Align Camarilla levels to 4h timeframe (using previous day's values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # --- 1d Indicators (HTF) ---
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 1d EMA34 (bullish trend) AND volume confirm
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND close < 1d EMA34 (bearish trend) AND volume confirm
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Pivot Point OR trend reverses (close < 1d EMA34)
            if (close[i] <= pp_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to Pivot Point OR trend reverses (close > 1d EMA34)
            if (close[i] >= pp_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals