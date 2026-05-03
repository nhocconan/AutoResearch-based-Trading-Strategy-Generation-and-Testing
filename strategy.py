#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4L4 breakout with 1d volume spike and 1w EMA200 trend filter.
# Long when price breaks above H4 level with 12h volume > 2.0x 20-period volume MA AND 1w close > 1w EMA200.
# Short when price breaks below L4 level with 12h volume > 2.0x 20-period volume MA AND 1w close < 1w EMA200.
# Exit when price returns to Pivot level or trend reverses.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla pivot levels provide precise intraday support/resistance, 1w EMA200 filters for higher-timeframe trend alignment,
# volume spike confirms institutional participation. Works in both bull and bear markets by only trading breakouts
# in the direction of the 1w trend when volume confirms.

name = "12h_Camarilla_H4L4_Breakout_1wEMA200_VolumeSpike_Session"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend direction
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.1*(high-low)/2
    # L4 = close - 1.1*(high-low)/2
    # Pivot = (high + low + close)/3
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    h4 = prev_close + 1.1 * camarilla_range / 2.0
    l4 = prev_close - 1.1 * camarilla_range / 2.0
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(volume_ma_12h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        
        # Volume spike condition: current 12h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_12h[i] * 2.0)
        
        # Trend conditions
        trend_up = close_val > ema_200_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_200_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: Price breaks above H4 level with volume spike AND 1w uptrend AND session
            if close_val > h4_aligned[i] and volume_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L4 level with volume spike AND 1w downtrend AND session
            elif close_val < l4_aligned[i] and volume_spike and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price returns to Pivot level OR trend reverses
            if close_val <= pivot_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price returns to Pivot level OR trend reverses
            if close_val >= pivot_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals