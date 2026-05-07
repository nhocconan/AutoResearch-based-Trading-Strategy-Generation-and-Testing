#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Regime
Hypothesis: Breakouts beyond Camarilla R1/S1 levels with 1-day trend alignment and volume confirmation capture strong momentum moves. Uses chop filter to avoid range-bound periods. Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend). Designed for low trade frequency (<400 total) to minimize fee drag.
"""
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Regime"
timeframe = "4h"
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
    
    # Calculate Camarilla levels from previous day
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 4.0  # R1 = close + 1.1*(high-low)/4
    s1 = prev_close - 1.1 * rang / 4.0  # S1 = close - 1.1*(high-low)/4
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    # Chop filter: avoid trading in high chop (>61.8)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((high14 - low14) / (atr14 * 14)) / np.log10(14)
    chop_filter = chop < 61.8  # Only trade when NOT choppy
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R1 + 1d uptrend + volume + low chop
            if close[i] > r1[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 + 1d downtrend + volume + low chop
            elif close[i] < s1[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to previous day's close (mean reversion)
            if position == 1:
                if close[i] <= prev_close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= prev_close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals