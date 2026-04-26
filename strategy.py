#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts with 1d EMA50 trend filter and choppiness regime (CHOP<61.8 = trending) + volume confirmation (1.5x median volume) yield high-probability entries. Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year (50-150 total over 4 years). Works in bull/bear via 1d trend and regime filters to avoid false breakouts in ranging markets.
"""

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
    
    # Calculate Camarilla levels for 12h (based on previous bar's range)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    
    # Volume confirmation: volume > 1.5x 30-period median (robust)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=30, min_periods=30).median().values
    volume_confirm = volume > (vol_median * 1.5)
    
    # Load 1d data for HTF trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d Choppiness Index (CHOP) for regime filter
    def choppiness_index(high, low, close, window=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        chop = np.zeros_like(close)
        for i in range(window, len(close)):
            if atr[i] > 0 and hh[i] > ll[i]:
                log_sum = np.log(atr[i] * window / (hh[i] - ll[i]))
                chop[i] = 100 * log_sum / np.log(window)
            else:
                chop[i] = 50.0
        return chop
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, window=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Regime: trending when CHOP < 61.8
    trending_regime = chop_1d_aligned < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30-period for volume median, 50 for EMA, 14 for CHOP)
    start_idx = max(30, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 with volume confirmation, 1d uptrend, and trending regime
        long_condition = (close[i] > r1[i]) and volume_confirm[i] and (close[i] > ema_50_1d_aligned[i]) and trending_regime[i]
        # Short logic: break below S1 with volume confirmation, 1d downtrend, and trending regime
        short_condition = (close[i] < s1[i]) and volume_confirm[i] and (close[i] < ema_50_1d_aligned[i]) and trending_regime[i]
        
        # Exit logic: opposite Camarilla level touch or trend reversal or ranging regime
        exit_long = (close[i] < s1[i]) or (close[i] < ema_50_1d_aligned[i]) or (not trending_regime[i])
        exit_short = (close[i] > r1[i]) or (close[i] > ema_50_1d_aligned[i]) or (not trending_regime[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0