# 25-04-2025 04:30:00 UTC
# 1d_1w_PriceAction_Volume_Regime
# Hypothesis: Daily price action (close > open) combined with weekly trend (weekly close > weekly open)
# and volume confirmation (volume > 1.5x 20-day average) filters for high-probability momentum trades.
# Works in bull markets via trend continuation and in bear markets via mean-reversion bounces
# at weekly support/resistance. Uses weekly timeframe for trend filter to avoid overtrading.
# Target: 15-25 trades/year on daily timeframe.

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
    
    # Get daily data for price action and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    volume_1d = df_1d['volume'].values
    
    # Daily bullish candle: close > open
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_filter = volume_1d > (vol_ma_20 * 1.5)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly trend: close > open (bullish trend)
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align signals to daily timeframe
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or \
           np.isnan(volume_filter_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or \
           np.isnan(weekly_bearish_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: daily bullish candle + volume + weekly bullish trend
        if daily_bullish_aligned[i] and volume_filter_aligned[i] and weekly_bullish_aligned[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        
        # Short entry: daily bearish candle + volume + weekly bearish trend
        elif daily_bearish_aligned[i] and volume_filter_aligned[i] and weekly_bearish_aligned[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        
        # Exit conditions: contrary signal or loss of weekly trend
        elif position == 1 and (daily_bearish_aligned[i] or not weekly_bullish_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (daily_bullish_aligned[i] or not weekly_bearish_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_PriceAction_Volume_Regime"
timeframe = "1d"
leverage = 1.0