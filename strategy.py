#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Regime_v1
Hypothesis: Use weekly Camarilla R1/S1 from weekly pivot points for breakout entries with 1d EMA34 trend filter and volume confirmation (>1.8x 20-period average). Add choppiness regime filter (CHOP > 61.8 = range, only mean-revert at extremes) to avoid whipsaws in bear markets. Target 12-37 trades/year on 12h timeframe. Weekly structure provides stronger support/resistance levels that work in both bull and bear markets via mean reversion at extremes.
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
    
    # Get weekly data for Camarilla calculation and daily data for trend filter
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 1 or len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate weekly OHLC for Camarilla pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels R1/S1 (based on previous weekly bar's range)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    camarilla_r1 = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 4
    camarilla_s1 = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 4
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly Camarilla levels and daily EMA to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending
    def choppiness_index(high, low, close, window=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = pd.Series(tr).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        hh_ll = max_high - min_low
        
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        chop = 100 * np.log10(sum_atr / np.maximum(hh_ll, 1e-10)) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    chop_range = chop > 61.8
    chop_trend = chop < 38.2
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need weekly Camarilla + daily EMA34 + volume MA + chop warmup
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            long_breakout = (close[i] > camarilla_r1_aligned[i]) and (close[i-1] > camarilla_r1_aligned[i-1])
            long_signal = long_breakout and trend_1d_uptrend and volume_spike[i] and not chop_range[i]
            
            short_breakout = (close[i] < camarilla_s1_aligned[i]) and (close[i-1] < camarilla_s1_aligned[i-1])
            short_signal = short_breakout and trend_1d_downtrend and volume_spike[i] and not chop_range[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
            if (close[i] < camarilla_s1_aligned[i] or not trend_1d_uptrend or chop_range[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            signals[i] = -0.25
            if (close[i] > camarilla_r1_aligned[i] or not trend_1d_downtrend or chop_range[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Regime_v1"
timeframe = "12h"
leverage = 1.0