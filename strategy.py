#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 12h volume confirmation (>1.5x 20-period average).
# Long when price breaks above R3 AND close > 1d EMA34 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below S3 AND close < 1d EMA34 (bearish trend) AND volume > 1.5x MA20.
# Exit when price retouches the 1d EMA34 level (mean reversion to trend) OR volume drops below average.
# Uses 1d HTF for trend to reduce noise and overtrading. Volume confirmation (>1.5x) reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 12h timeframe.
# Camarilla pivot levels provide high-probability reversal/breakout levels intraday.
# EMA34 on 1d filter ensures we only trade with the higher timeframe trend.
# Volume spike confirms institutional participation.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_12hVolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # Camarilla pivot levels calculation (based on previous day's OHLC)
    # We need to calculate daily pivot first, then derive Camarilla levels
    # For 12h timeframe, we use 1d OHLC to calculate Camarilla levels for the 12h chart
    
    # --- 1d Indicators (HTF) for trend and Camarilla calculation ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla calculation
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: 
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # R2 = close + (high - low) * 1.1/6
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # S2 = close - (high - low) * 1.1/6
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    
    # We need previous day's OHLC to calculate today's Camarilla levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_s3 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- 12h Indicators (LTF) continued ---
    # 12h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 1d EMA34 (bullish trend) AND volume confirm
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND close < 1d EMA34 (bearish trend) AND volume confirm
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retouches 1d EMA34 (mean reversion to trend) OR volume drops below average
            if (close[i] <= ema_34_1d_aligned[i] or 
                not volume_confirm_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retouches 1d EMA34 (mean reversion to trend) OR volume drops below average
            if (close[i] >= ema_34_1d_aligned[i] or 
                not volume_confirm_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals