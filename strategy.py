#!/usr/bin/env python3
# Hypothesis: 4h timeframe with daily ATR-based breakout and 12h trend filter.
# Uses daily ATR(14) to define breakout channels (upper/lower bands) and 12h EMA50 for trend direction.
# ATR-based breakouts adapt to volatility, reducing false signals in low-volatility periods.
# 12h trend filter ensures trades align with higher timeframe momentum, improving win rate in both bull and bear markets.
# Target: 80-150 total trades over 4 years (20-38/year) with size 0.25.

name = "4h_ATR_Breakout_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate daily ATR(14) for volatility-based breakout channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation
    prev_close = np.roll(df_1d['close'], 1)
    prev_close[0] = df_1d['close'].iloc[0]  # First value
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - prev_close)
    tr3 = np.abs(df_1d['low'] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate breakout channels: upper/lower bands based on ATR
    # Using 20-period lookback for channel calculation (adaptive to recent volatility)
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    upper_band = df_1d['close'].values + 1.5 * atr_ma
    lower_band = df_1d['close'].values - 1.5 * atr_ma
    
    # Align daily bands to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Breakout conditions: price breaks above upper band or below lower band
    breakout_up = close > upper_band_aligned
    breakout_down = close < lower_band_aligned
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    trend_up = close > ema_50_12h_aligned
    trend_down = close < ema_50_12h_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume (balanced to avoid overtrading)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper band + 12h uptrend + volume filter
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band + 12h downtrend + volume filter
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to mid-band or trend reversal
            mid_band = (upper_band_aligned[i] + lower_band_aligned[i]) / 2
            if close[i] <= mid_band or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to mid-band or trend reversal
            mid_band = (upper_band_aligned[i] + lower_band_aligned[i]) / 2
            if close[i] >= mid_band or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals