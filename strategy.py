#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with 1w EMA trend filter and volume confirmation.
# Long when price breaks above weekly high in weekly uptrend with volume surge.
# Short when price breaks below weekly low in weekly downtrend with volume surge.
# Uses weekly EMA(34) for trend direction and daily volume > 2x 20-day EMA for confirmation.
# Designed for low trade frequency (7-25/year) to minimize fee fraud and capture sustained moves.

name = "1d_WeeklyDonchian_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1w[1:] > ema_34_1w[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with weekly index
    
    # Weekly Donchian channels (20-period)
    high_roll = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    high_roll_aligned = align_htf_to_ltf(prices, df_1w, high_roll)
    low_roll_aligned = align_htf_to_ltf(prices, df_1w, low_roll)
    
    # Volume confirmation: daily volume > 2.0x 20-day EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(high_roll_aligned[i]) or
            np.isnan(low_roll_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above weekly high in weekly uptrend with volume
            if (trend_up_aligned[i] > 0.5 and  # Weekly uptrend
                close[i] > high_roll_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below weekly low in weekly downtrend with volume
            elif (trend_up_aligned[i] <= 0.5 and  # Weekly downtrend
                  close[i] < low_roll_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below weekly low or trend turns down
            if close[i] < low_roll_aligned[i] or trend_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above weekly high or trend turns up
            if close[i] > high_roll_aligned[i] or trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals