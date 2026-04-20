#!/usr/bin/env python3
# 1d_1w_Camarilla_R1_S1_Breakout_VolumeATRFilter
# Hypothesis: Daily chart with weekly trend filter - trade breakouts of weekly-derived Camarilla R1/S1 levels on daily timeframe.
# Uses weekly trend direction (price above/below weekly EMA20) to determine bias, then trades breakouts of daily R1/S1 with volume confirmation.
# Works in bull markets by buying R1 breaks in uptrend; in bear markets by selling S1 breaks in downtrend.
# Volume filter ensures institutional participation, ATR filter avoids low-volatility false breakouts.
# Target: 15-30 trades/year to minimize fee drag on daily timeframe.

name = "1d_1w_Camarilla_R1_S1_Breakout_VolumeATRFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly trend filter: EMA20
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend_up = weekly_close > weekly_ema20  # True when in weekly uptrend
    
    # Align weekly trend to daily timeframe
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for today's pivot (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First period uses current values
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12
    
    # Align daily Camarilla levels to daily timeframe (already aligned but using for consistency)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.5)
    
    # ATR filter: avoid low-volatility breakouts
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_filter = atr > (atr_ma * 0.7)  # Only trade when volatility is above 70% of its 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(atr_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + weekly uptrend + volume + volatility confirmation
            if (close[i] > r1_aligned[i] and weekly_trend_up_aligned[i] > 0.5 and 
                volume_filter[i] and atr_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + weekly downtrend + volume + volatility confirmation
            elif (close[i] < s1_aligned[i] and weekly_trend_up_aligned[i] < 0.5 and 
                  volume_filter[i] and atr_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal) or weekly trend turns down
            if close[i] < s1_aligned[i] or weekly_trend_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal) or weekly trend turns up
            if close[i] > r1_aligned[i] or weekly_trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals