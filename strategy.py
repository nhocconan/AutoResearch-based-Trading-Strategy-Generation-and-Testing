#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d1w_camarilla_volume_trend
# Uses daily and weekly Camarilla pivot levels on 12h chart.
# Long when price breaks above weekly R4 with volume confirmation (volume > 1.5x 20-period avg) and daily trend up (close > EMA20).
# Short when price breaks below weekly S4 with volume confirmation and daily trend down (close < EMA20).
# Exits when price returns to weekly close (mean reversion).
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drift.
# Works in both bull and bear via trend filter and avoids false breakouts in chop.

name = "12h_1d1w_camarilla_volume_trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift by 1 to use previous week's data
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev = np.roll(close_1w, 1)
    
    # First value will be invalid due to roll, set to NaN
    high_1w_prev[0] = np.nan
    low_1w_prev[0] = np.nan
    close_1w_prev[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_R4 = close_1w_prev + 1.5 * (high_1w_prev - low_1w_prev)
    camarilla_S4 = close_1w_prev - 1.5 * (high_1w_prev - low_1w_prev)
    # Exit level: previous week's close
    camarilla_exit = close_1w_prev
    
    # Align weekly Camarilla levels to 12h timeframe
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R4)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S4)
    camarilla_exit_aligned = align_htf_to_ltf(prices, df_1w, camarilla_exit)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Volume confirmation: volume > 1.5 * 20-period average (daily timeframe)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm_1d = volume_1d > (vol_ma_1d * 1.5)
    
    # Align daily volume confirmation to 12h timeframe
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_R4_aligned[i]) or np.isnan(camarilla_S4_aligned[i]) or 
            np.isnan(camarilla_exit_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly R4 with volume confirmation and uptrend
        if (close[i] > camarilla_R4_aligned[i] and vol_confirm_aligned[i] and 
            close[i] > ema20_1w_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly S4 with volume confirmation and downtrend
        elif (close[i] < camarilla_S4_aligned[i] and vol_confirm_aligned[i] and 
              close[i] < ema20_1w_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to previous week's close (mean reversion)
        elif position == 1 and close[i] <= camarilla_exit_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= camarilla_exit_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals