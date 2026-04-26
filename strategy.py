#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: On 12h timeframe, use Camarilla R1/S1 from 1d pivot points for breakout entries with 1d trend filter (close > 1d EMA34) and volume confirmation (>2.0x 20-period average). Add choppiness regime filter (CHOP > 61.8 = range, only mean-revert at extremes) to avoid whipsaws in bear markets. Target: 12-37 trades/year. Uses R1/S1 levels (proven edge) with volume/trend confirmation to reduce false breakouts while maintaining edge in both bull and bear regimes.
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels R1/S1 (based on previous 1d bar's range)
    # Camarilla R1 = close + 1.1*(high - low)/4
    # Camarilla S1 = close - 1.1*(high - low)/4
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first value to NaN (no previous bar)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r1 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 4
    camarilla_s1 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 4
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending
    def choppiness_index(high, low, close, window=14):
        # True range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for first bar
        
        # ATR = smoothed TR (using Wilder's smoothing = EMA with alpha=1/window)
        atr = pd.Series(tr).ewm(alpha=1/window, adjust=False, min_periods=window).mean().values
        
        # Max(high) - Min(low) over window
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        hh_ll = max_high - min_low
        
        # CHOP = 100 * log10(sum(atr)/hh_ll) / log10(window)
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        chop = 100 * np.log10(sum_atr / np.maximum(hh_ll, 1e-10)) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    chop_range = chop > 61.8  # range regime
    chop_trend = chop < 38.2  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1d EMA34) + volume MA + chop warmup
    start_idx = max(34, 20, 14)  # 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume spike + NOT in strong range (avoid false breakouts)
            # Require confirmation: price outside bands for 2 consecutive bars
            long_breakout = (close[i] > camarilla_r1_aligned[i]) and (close[i-1] > camarilla_r1_aligned[i-1])
            long_signal = long_breakout and trend_1d_uptrend and volume_spike[i] and not chop_range[i]
            
            # Short: price breaks below S1 + 1d downtrend + volume spike + NOT in strong range
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
            # Hold long
            signals[i] = 0.25
            # Exit: price touches S1 OR 1d trend turns down OR chop becomes strong range (revert to mean)
            if (close[i] < camarilla_s1_aligned[i] or not trend_1d_uptrend or chop_range[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R1 OR 1d trend turns up OR chop becomes strong range
            if (close[i] > camarilla_r1_aligned[i] or not trend_1d_downtrend or chop_range[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0