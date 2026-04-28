#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R reversal with 1w EMA34 trend filter and volume spike confirmation.
# Targets 7-25 trades/year (30-100 total) by using extreme %R levels (<-80 for long, >-20 for short)
# combined with 1w EMA34 trend filter and volume confirmation (>2.0x 20-bar average).
# Williams %R identifies overbought/oversold conditions that often reverse in crypto markets.
# 1w EMA34 provides medium-term trend filter to avoid counter-trend trades.
# Discrete position sizing (±0.25) minimizes fee churn while maintaining adequate exposure.
# Works in both bull and bear markets via trend filter + mean reversion logic.

name = "1d_WilliamsR_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and 1w data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 14 or len(df_1w) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Williams %R extreme conditions
        williams_r_oversold = williams_r_aligned[i] < -80  # Oversold
        williams_r_overbought = williams_r_aligned[i] > -20  # Overbought
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = williams_r_oversold and price_above_ema and vol_confirm
        short_entry = williams_r_overbought and price_below_ema and vol_confirm
        
        # Exit conditions: Williams %R returns to neutral territory
        long_exit = williams_r_aligned[i] > -50  # Exit long when %R > -50
        short_exit = williams_r_aligned[i] < -50  # Exit short when %R < -50
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals