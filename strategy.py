#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Camarilla pivot level breaks (R1/S1) with 12h trend confirmation and volume surge.
# Long when price breaks above R1 in a 12h uptrend with volume surge.
# Short when price breaks below S1 in a 12h downtrend with volume surge.
# Exits when price returns to the Camarilla mid-point (C) or trend reverses.
# Designed for 4h timeframe to work in both bull and bear markets by using 12h trend filter.
# Uses discrete position sizing (0.25) to limit turnover and fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation (using 12h data as proxy for daily)
    # Since we don't have 1d data directly, we'll use the last complete 12h bar's data
    # We need to look back to get the previous day's range
    # For simplicity, we'll use the 12h data to approximate daily range
    # In practice, Camarilla uses prior day's OHLC, but we approximate with 2-period 12h aggregation
    # Alternative: get 1d data if available, but per rules we use available TFs
    
    # Calculate Camarilla levels from previous day's range
    # We'll use 12h data and assume 2 bars = 1 day for approximation
    # Better: resample 12h to daily? No, per rules: use get_htf_data for actual TF
    # Since we can't get 1d directly in this context, we'll use the prior 12h bar's high/low
    # This is an approximation but follows the spirit of the strategy
    
    # Actually, let's use the 12h data to get the prior session's range
    # We'll shift the 12h data by 1 to get the previous period
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla for each 12h bar using that bar's OHLC (standard approach)
    # Camarilla levels:
    # H = high, L = low, C = close
    # R4 = C + (H-L)*1.5/2
    # R3 = C + (H-L)*1.25/2
    # R2 = C + (H-L)*1.16/2
    # R1 = C + (H-L)*1.0833/2
    # S1 = C - (H-L)*1.0833/2
    # S2 = C - (H-L)*1.16/2
    # S3 = C - (H-L)*1.25/2
    # S4 = C - (H-L)*1.5/2
    # We only need R1 and S1 for entry, and C (pivot) for exit
    
    H = high_12h
    L = low_12h
    C = close_12h
    range_hl = H - L
    
    R1 = C + range_hl * 1.0833 / 2
    S1 = C - range_hl * 1.0833 / 2
    Pivot = (H + L + C) / 3  # Standard pivot, though Camarilla uses close as pivot
    
    # Camarilla actually uses close as the pivot point for R/S calculations
    # But for exit we'll use the Camarilla pivot point which is (H+L+C)/3
    
    # Calculate the levels
    R1 = C + (H - L) * 1.0833 / 2
    S1 = C - (H - L) * 1.0833 / 2
    Camarilla_Pivot = (H + L + C) / 3  # This is the actual Camarilla pivot
    
    # Align Camarilla levels to 4h
    R1_4h = align_htf_to_ltf(prices, df_12h, R1)
    S1_4h = align_htf_to_ltf(prices, df_12h, S1)
    Pivot_4h = align_htf_to_ltf(prices, df_12h, Camarilla_Pivot)
    
    # 12h EMA34 for trend (smoother than EMA50 for less whipsaw)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_slope = ema_34_12h - np.roll(ema_34_12h, 1)
    ema_34_12h_slope[0] = 0
    ema_34_12h_slope = pd.Series(ema_34_12h_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    ema_34_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_slope)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 (34) and smoothing (3)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(Pivot_4h[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(ema_34_12h_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 12h EMA34 slope
        uptrend = ema_34_12h_slope_aligned[i] > 0
        downtrend = ema_34_12h_slope_aligned[i] < 0
        
        if position == 0:
            # Look for breakout with volume surge
            if uptrend and vol_surge[i]:
                # Long: price breaks above R1
                if close[i] > R1_4h[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: price breaks below S1
                if close[i] < S1_4h[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to pivot OR trend reverses
                if close[i] <= Pivot_4h[i] or downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot OR trend reverses
                if close[i] >= Pivot_4h[i] or uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals