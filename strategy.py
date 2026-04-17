#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with Volume Spike and ADX Trend Filter
Enters long when price breaks above R1 with volume > 2x 20-period average and ADX > 25 (uptrend)
Enters short when price breaks below S1 with volume > 2x 20-period average and ADX > 25 (downtrend)
Exits when price returns to the pivot point (PP) or volume drops below average.
Designed for 12h timeframe to capture institutional breakout moves with volume confirmation.
Target: 15-30 trades/year (60-120 total over 4 years) by requiring confluence of breakout, volume, and trend.
Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Camarilla Pivot Levels (from previous day) ===
    df_1d = get_htf_data(prices, '1d')
    # Need previous day's OHLC for today's Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for today
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (prev_high + prev_low + prev_close) / 3
    # Range = High - Low
    range_val = prev_high - prev_low
    # Resistance 1 (R1) = Close + (Range * 1.1 / 12)
    r1 = prev_close + (range_val * 1.1 / 12)
    # Support 1 (S1) = Close - (Range * 1.1 / 12)
    s1 = prev_close - (range_val * 1.1 / 12)
    
    # Align to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily Volume Spike (2x 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === Weekly ADX Trend Filter (ADX > 25) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement (+DM, -DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1w = wilders_smoothing(tr, period)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, period) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, period) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, period)
    
    # Handle division by zero and NaN
    adx_1w = np.where((plus_di_1w + minus_di_1w) == 0, 0, adx_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for all calculations
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 2x 20-day average
        vol_today_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_today_aligned[i] > vol_ma_20_1d_aligned[i] * 2.0
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_1w_aligned[i] > 25
        
        # Breakout conditions
        breakout_up = high[i] > r1_aligned[i]   # Price breaks above R1
        breakdown_down = low[i] < s1_aligned[i] # Price breaks below S1
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish breakout above R1 with volume confirmation and trend
            if breakout_up and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish breakdown below S1 with volume confirmation and trend
            elif breakdown_down and vol_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to pivot point or volume fails
        elif position == 1:
            # Exit long: price returns to pivot point or volume confirmation fails
            if low[i] <= pp_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point or volume confirmation fails
            if high[i] >= pp_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0