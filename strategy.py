#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold) with 1w bullish trend (close > EMA50) and volume > 1.3x 20-period volume EMA
# Short when Williams %R(14) crosses below -20 (overbought) with 1w bearish trend (close < EMA50) and volume > 1.3x 20-period volume EMA
# Uses 1w EMA50 for major trend filter to reduce whipsaw and align with higher timeframe momentum.
# Williams %R provides mean-reversion signals in ranging markets while trend filter ensures we trade with the higher timeframe direction.
# Volume confirmation (1.3x) adds validity to breakouts from extreme levels.
# Targets 7-25 trades/year on 1d timeframe by requiring confluence of three filters.

name = "1d_WilliamsR_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1w = close_1w > ema_50_1w
    trend_bearish_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate Williams %R(14) from 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_long_signal = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_short_signal = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.3)  # Volume at least 1.3x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or
            np.isnan(williams_long_signal[i]) or np.isnan(williams_short_signal[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 AND 1w bullish trend AND volume spike
            if (williams_long_signal[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 AND 1w bearish trend AND volume spike
            elif (williams_short_signal[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (profit taking) OR 1w trend turns bearish
            if (williams_r[i] < -50 and np.roll(williams_r, 1)[i] >= -50) or \
               trend_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (profit taking) OR 1w trend turns bullish
            if (williams_r[i] > -50 and np.roll(williams_r, 1)[i] <= -50) or \
               trend_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals