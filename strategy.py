#!/usr/bin/env python3
# 4h_ThreeBand_Squeeze_Breakout
# Hypothesis: Combines Bollinger Band squeeze, Donchian breakout, and 1-day trend to capture explosive moves in both bull and bear markets.
# Bollinger Band width below 20th percentile indicates low volatility (squeeze).
# Breakout occurs when price closes above/below Donchian(20) channel.
# Direction confirmed by 1-day EMA50 slope to avoid counter-trend trades.
# Volume surge (>1.5x 20-period average) filters false breakouts.
# Works in bull markets (catching breakouts) and bear markets (catching breakdowns).
# Uses fixed position size of 0.25 to manage risk and reduce fee churn.

name = "4h_ThreeBand_Squeeze_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Bollinger Band (20, 2) width ---
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_width_values = bb_width.values
    
    # Bollinger Band width percentile (20-period lookback)
    bb_width_percentile = pd.Series(bb_width_values).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # --- Donchian Channel (20) ---
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1-day EMA50 for trend direction ---
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_1d_slope[0] = 0
    ema_50_1d_slope = pd.Series(ema_50_1d_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_slope)
    
    # --- Volume confirmation (volume > 1.5x 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for BB (20), Donchian (20), EMA50 slope (50+3)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bb_width_percentile[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_slope_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB width below 20th percentile
        squeeze = bb_width_percentile[i] < 0.20
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Trend direction from 1-day EMA50 slope
        uptrend = ema_50_1d_slope_aligned[i] > 0
        downtrend = ema_50_1d_slope_aligned[i] < 0
        
        if position == 0:
            if squeeze and breakout_up and vol_surge[i] and uptrend:
                # Long: squeeze breakout up + volume surge + 1-day uptrend
                signals[i] = 0.25
                position = 1
            elif squeeze and breakout_down and vol_surge[i] and downtrend:
                # Short: squeeze breakout down + volume surge + 1-day downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price closes below Donchian mid or trend reverses
                donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] < donchian_mid or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above Donchian mid or trend reverses
                donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] > donchian_mid or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals