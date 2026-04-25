#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v2
Hypothesis: Daily Camarilla R3/S3 breakouts with 1-week ADX(14)>25 trend filter and volume confirmation (2x 20-day avg).
Only trade breakouts aligned with strong weekly trend to avoid whipsaws. Volume confirms institutional participation.
Designed for 1d timeframe targeting 10-20 trades/year. Works in bull/bear by following 1w ADX trend.
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
    
    # Get 1w data for HTF trend filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1w data for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 1d timeframe (1-week lagged for completed bar)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=1)
    plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di, additional_delay_bars=1)
    minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di, additional_delay_bars=1)
    
    # Calculate Camarilla levels on 1d data (using previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3/S3 for breakouts
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First period
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range
    s3 = prev_close - 1.1 * camarilla_range
    
    # Volume confirmation: 2x 20-day average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Camarilla (1) and ADX (50)
    start_idx = max(1, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3[i]) or 
            np.isnan(s3[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(plus_di_aligned[i]) or
            np.isnan(minus_di_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        trend_bullish = strong_trend and (plus_di_aligned[i] > minus_di_aligned[i])
        trend_bearish = strong_trend and (minus_di_aligned[i] > plus_di_aligned[i])
        
        if position == 0:
            # Look for breakout signals with volume confirmation and strong trend alignment
            long_signal = (close[i] > r3[i]) and volume_spike[i] and trend_bullish
            short_signal = (close[i] < s3[i]) and volume_spike[i] and trend_bearish
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price breaks below S3 or trend weakens (ADX < 20)
            exit_signal = (close[i] < s3[i]) or (adx_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above R3 or trend weakens (ADX < 20)
            exit_signal = (close[i] > r3[i]) or (adx_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v2"
timeframe = "1d"
leverage = 1.0