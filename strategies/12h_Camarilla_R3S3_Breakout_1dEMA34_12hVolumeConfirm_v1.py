#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and 12h volume confirmation (>1.8x 20-period average).
# Long when price breaks above R3 AND close > 1d EMA34 (bullish trend) AND volume > 1.8x 20-period average.
# Short when price breaks below S3 AND close < 1d EMA34 (bearish trend) AND volume > 1.8x 20-period average.
# Exit when price retests the 1d EMA34 level (mean reversion to trend) or opposite Camarilla level touched.
# Uses 1d HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (1.8x) reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 12h timeframe.
# Camarilla pivot levels provide high-probability reversal/breakout levels, effective in both bull and bear markets when combined with HTF trend filter.

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
    # 12h volume confirmation: > 1.8x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.8 * vol_ma_20)
    
    # --- 12h Camarilla Pivot Levels (R3, S3) ---
    # Calculate from previous 12h bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r3 = pivot + (range_ * 1.1 / 2.0)  # R3 = pivot + (high-low)*1.1/2
    s3 = pivot - (range_ * 1.1 / 2.0)  # S3 = pivot - (high-low)*1.1/2
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3[i]) or
            np.isnan(s3[i]) or
            np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 1d EMA34 (bullish trend) AND volume confirm
            if (close[i] > r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND close < 1d EMA34 (bearish trend) AND volume confirm
            elif (close[i] < s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests 1d EMA34 (mean reversion to trend) OR touches S3 (opposite level)
            if (close[i] <= ema_34_1d_aligned[i] or 
                close[i] < s3[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests 1d EMA34 (mean reversion to trend) OR touches R3 (opposite level)
            if (close[i] >= ema_34_1d_aligned[i] or 
                close[i] > r3[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals