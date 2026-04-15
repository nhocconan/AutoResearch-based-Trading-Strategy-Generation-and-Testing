#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Candlestick Reversal with Weekly Trend Filter and Volume Spike
# Uses bullish/bearish engulfing patterns on daily timeframe as entry signals.
# Filters by weekly trend (price above/below weekly EMA20) to trade with the higher timeframe trend.
# Requires volume spike (2x 20-day average) for confirmation.
# Designed to work in both bull and bear markets by following weekly trend direction.
# Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for engulfing patterns
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(df_1d, df_1w, ema_20_1w)
    
    # Calculate bullish and bearish engulfing patterns on daily data
    # Bullish engulfing: current candle engulfs previous bearish candle
    bullish_engulf = (close_1d > open_1d) & (open_1d < close_1d) & \
                     (close_1d > open_1d) & (open_1d > close_1d) & \
                     (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1))
    # Actually: current bullish candle body completely engulfs previous bearish candle body
    bullish_engulf = (close_1d > open_1d) & (open_1d < close_1d) & \
                     (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1)) & \
                     (close_1d - open_1d) > (open_1d.shift(1) - close_1d.shift(1))
    
    # Bearish engulfing: current candle engulfs previous bullish candle
    bearish_engulf = (close_1d < open_1d) & (open_1d > close_1d) & \
                     (close_1d < open_1d.shift(1)) & (open_1d > close_1d.shift(1)) & \
                     (open_1d - close_1d) > (close_1d.shift(1) - open_1d.shift(1))
    
    # Align engulfing signals to 1d timeframe (no additional delay needed as engulfing is confirmed at daily close)
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bullish_engulf.astype(float))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bearish_engulf.astype(float))
    
    # Calculate volume spike: current volume > 2x 20-day average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * volume_ma_20)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    # Start from index 20 to ensure volume MA is valid
    start_idx = 20
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_engulf_aligned[i]) or np.isnan(bearish_engulf_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            continue
        
        # Long entry: bullish engulfing + price above weekly EMA20 + volume spike
        if (bullish_engulf_aligned[i] > 0.5 and
            close[i] > ema_20_1w_aligned[i] and
            volume_spike[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish engulfing + price below weekly EMA20 + volume spike
        elif (bearish_engulf_aligned[i] > 0.5 and
              close[i] < ema_20_1w_aligned[i] and
              volume_spike[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite engulfing signal or price crosses weekly EMA20 in opposite direction
        elif position == 1 and (bearish_engulf_aligned[i] > 0.5 or close[i] < ema_20_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_engulf_aligned[i] > 0.5 or close[i] > ema_20_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Engulfing_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0