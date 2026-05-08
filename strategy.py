#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Donchian breakout with 1d trend filter and volume confirmation.
# Long when price breaks above weekly high in uptrend with volume surge.
# Short when price breaks below weekly low in downtrend with volume surge.
# Uses 1d EMA(34) for trend direction and 1d ATR(14) for dynamic breakout levels.
# Designed for low trade frequency (15-25/year) to minimize fee drag and capture sustained moves.
# Weekly Donchian provides strong structural breaks, 1d trend filters false breakouts,
# volume confirms institutional participation.

name = "6h_WeeklyDonchian_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # 1d ATR(14) for volatility
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_1d[0] - close_1d[0]  # First value
    low_close[0] = low_1d[0] - close_1d[0]    # First value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian(20): highest high and lowest low of past 20 weekly candles
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 6h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    # Align weekly Donchian levels to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Volume confirmation: 6h volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above weekly high in uptrend with volume
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] > high_20_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below weekly low in downtrend with volume
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] < low_20_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below weekly low or trend turns down
            if close[i] < low_20_aligned[i] or trend_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above weekly high or trend turns up
            if close[i] > high_20_aligned[i] or trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals