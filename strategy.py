#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Weekly ATR Breakout + Volume + ADX Trend
# Hypothesis: Combines weekly ATR-based breakout for trend detection with daily volume confirmation
# and ADX trend strength filter. Designed to capture major trend moves while avoiding whipsaws
# in ranging markets. Weekly timeframe reduces noise, 12h provides timely entries. Works in both
# bull and bear markets by following established trends with volatility-adjusted breaks.
# Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
name = "12h_weekly_atr_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) on weekly data
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    tr1 = high_w - low_w
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_w = np.zeros_like(tr)
    atr_w[13] = np.mean(tr[0:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_w[i] = (atr_w[i-1] * 13 + tr[i]) / 14
    
    # Calculate upper and lower bands: close ± 2.5 * ATR
    upper_w = close_w + 2.5 * atr_w
    lower_w = close_w - 2.5 * atr_w
    
    # Align weekly bands to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_weekly, upper_w)
    lower_12h = align_htf_to_ltf(prices, df_weekly, lower_w)
    
    # Get daily data for volume and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Daily volume average
    daily_volume = df_daily['volume'].values
    vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h = align_htf_to_ltf(prices, df_daily, vol_ma)
    
    # Daily ADX(14) for trend strength
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_daily[0] = tr1[0]
    
    # Calculate +DM and -DM
    up_move = daily_high - np.roll(daily_high, 1)
    down_move = np.roll(daily_low, 1) - daily_low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    def smooth_series(x, period):
        result = np.zeros_like(x)
        if len(x) < period:
            return result
        result[period-1] = np.mean(x[0:period])
        for i in range(period, len(x)):
            result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr_daily = smooth_series(tr_daily, 14)
    plus_di = 100 * smooth_series(plus_dm, 14) / atr_daily
    minus_di = 100 * smooth_series(minus_dm, 14) / atr_daily
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_series(dx, 14)
    
    # Align daily indicators to 12h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_daily, vol_ma)
    adx_12h = align_htf_to_ltf(prices, df_daily, adx)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-day average
        vol_filter = volume[i] > (vol_ma_aligned[i] * 1.3)
        
        # ADX filter: trend strength > 20
        adx_filter = adx_12h[i] > 20
        
        if position == 1:  # Long position
            # Exit: price closes below lower band or trend weakens
            if close[i] < lower_12h[i] or adx_12h[i] < 15:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above upper band or trend weakens
            if close[i] > upper_12h[i] or adx_12h[i] < 15:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume and ADX filters
            if vol_filter and adx_filter:
                # Long: price breaks above upper band
                if close[i] > upper_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower band
                elif close[i] < lower_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals