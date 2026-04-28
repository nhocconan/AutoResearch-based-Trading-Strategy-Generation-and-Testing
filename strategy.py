#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA200 trend + volume confirmation
# Donchian channel breakouts capture strong momentum moves.
# 1w EMA200 filter ensures alignment with long-term trend, reducing false signals in ranging markets.
# Volume confirmation (>1.5x 20-bar average) validates breakout strength.
# Exits on retracement to midpoint of Donchian channel or opposite breakout.
# Designed for 12h timeframe to limit trade frequency (target: 50-150 total trades over 4 years).
# Works in both bull and bear markets by requiring trend alignment.
# Uses discrete position sizing (0.25) to manage drawdown and reduce fee churn.

name = "12h_Donchian20_Breakout_1wEMA200_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Donchian channels (20-period) from 12h data
    # We need at least 20 periods, so we'll use a rolling window on the 12h data itself
    # Since we're on 12h timeframe, we can calculate directly from prices
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Upper channel: 20-period high
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    # Middle channel: average of upper and lower
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA200 trend filter
        ema_trend_up = close[i] > ema_200_1w_aligned[i]
        ema_trend_down = close[i] < ema_200_1w_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > upper Donchian, 1w EMA200 uptrend, volume confirm
            if price > donchian_high[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < lower Donchian, 1w EMA200 downtrend, volume confirm
            elif price < donchian_low[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to middle or below lower channel
            if price < donchian_mid[i] or price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to middle or above upper channel
            if price > donchian_mid[i] or price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals