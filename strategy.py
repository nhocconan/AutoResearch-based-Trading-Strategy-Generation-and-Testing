#!/usr/bin/env python3
"""
1h_Bollinger_Bandwidth_Breakout_4hTrend_Volume
Hypothesis: Use Bollinger Bandwidth to identify volatility contractions (squeeze) followed by breakouts.
Only take breakouts in the direction of 4h EMA50 trend to avoid counter-trend trades.
Volume spike confirms breakout legitimacy.
This strategy works in both bull and bear markets by following the 4h trend direction.
Designed for low trade frequency (15-30/year) with high win rate by requiring volatility squeeze,
trend alignment, and volume confirmation.
"""

name = "1h_Bollinger_Bandwidth_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) for squeeze detection
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bandwidth = (upper_band - lower_band) / sma_20  # Normalized bandwidth
    
    # Bandwidth percentile (20-period) to identify squeeze
    bandwidth_series = pd.Series(bandwidth)
    bandwidth_percentile = bandwidth_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    squeeze = bandwidth_percentile < 10  # Bottom 10% = squeeze
    
    # Breakout conditions: price breaks above/below Bollinger Bands
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    trend_up = close > ema_50_4h_aligned
    trend_down = close < ema_50_4h_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(squeeze[i]) or np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or np.isnan(volume_filter[i]) or
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout up + 4h uptrend + volume spike + session
            if squeeze[i-1] and breakout_up[i] and trend_up[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: squeeze breakout down + 4h downtrend + volume spike + session
            elif squeeze[i-1] and breakout_down[i] and trend_down[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle band or trend reversal
            if close[i] < sma_20[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to middle band or trend reversal
            if close[i] > sma_20[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals