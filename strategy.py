#!/usr/bin/env python3
# 6h_1d_1w_WilliamsFractal_Trend_Volume
# Hypothesis: Use 1w Williams Fractals to identify long-term support/resistance levels.
# Breakout above weekly bearish fractal in uptrend (1d EMA34 slope > 0) with volume surge (2x 20-period avg) goes long.
# Breakdown below weekly bullish fractal in downtrend (1d EMA34 slope < 0) with volume surge goes short.
# Williams Fractals require 2-bar confirmation (additional_delay_bars=2) to avoid look-ahead.
# Designed for low trade frequency (15-30/year) to minimize fee drag, works in both bull and bear markets by following the weekly trend.

name = "6h_1d_1w_WilliamsFractal_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for Williams Fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Williams Fractals (requires 5-bar window: t-2, t-1, t, t+1, t+2)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values
    )
    # Bearish fractal: high[t] is highest among [t-2, t-1, t, t+1, t+2]
    # Bullish fractal: low[t] is lowest among [t-2, t-1, t, t+1, t+2]
    # These are confirmed only after 2 additional bars close, so use additional_delay_bars=2
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slope_34_1d = np.diff(ema_34_1d, prepend=ema_34_1d[0])  # slope = today - yesterday
    ema_slope_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_34_1d)
    
    # ATR for volatility and trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_slope_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from daily EMA34 slope
        bullish_trend = ema_slope_34_1d_aligned[i] > 0
        bearish_trend = ema_slope_34_1d_aligned[i] < 0
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above weekly bearish fractal in bullish trend with volume surge
            if close[i] > bearish_fractal_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: breakdown below weekly bullish fractal in bearish trend with volume surge
            elif close[i] < bullish_fractal_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 2.5*ATR from highest high
                if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 2.5*ATR from lowest low
                if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals