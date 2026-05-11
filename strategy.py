#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breaks above Camarilla R1 level or below S1 level on 4h with 1d trend direction and volume surge.
# Works in bull markets by buying breakouts above R1 in uptrends.
# Works in bear markets by selling breakouts below S1 in downtrends.
# Uses 1d EMA34 for trend and volume confirmation to avoid false breakouts.
# Target: 20-50 trades per year per symbol.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend direction ---
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_slope = ema_34_1d - np.roll(ema_34_1d, 1)
    ema_34_1d_slope[0] = 0
    ema_34_1d_slope = pd.Series(ema_34_1d_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_slope)
    
    # --- 4h Camarilla levels (based on previous day's OHLC) ---
    # Note: Camarilla levels are calculated from previous day's range
    # We'll use the 1d data to compute levels for the current 4h period
    # Camarilla: H = high, L = low, C = close
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    
    # Calculate daily range from 1d data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels for each day
    camarilla_R1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_S1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 (34) and smoothing (3)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_slope_aligned[i]) or
            np.isnan(camarilla_R1_aligned[i]) or
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1d EMA34 slope
        uptrend = ema_34_1d_slope_aligned[i] > 0
        downtrend = ema_34_1d_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 1d uptrend + volume surge + price breaks above R1
                if close[i] > camarilla_R1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 1d downtrend + volume surge + price breaks below S1
                if close[i] < camarilla_S1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 1d trend turns down OR price crosses below S1 (reversal)
                if downtrend or close[i] < camarilla_S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 1d trend turns up OR price crosses above R1 (reversal)
                if uptrend or close[i] > camarilla_R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals