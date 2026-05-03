#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R(14) mean reversion with 1d trend filter (EMA34) and volume confirmation.
# Long when Williams %R < -80 (oversold) in 1d uptrend (price > EMA34) with volume > 1.2x 20-period MA.
# Short when Williams %R > -20 (overbought) in 1d downtrend (price < EMA34) with volume confirmation.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
# Williams %R is effective in ranging/bear markets (2025+) and captures mean reversion after extreme moves.

name = "12h_WilliamsR14_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 12h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.2 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        wr = williams_r[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND 1d uptrend AND volume spike
            if wr < -80 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND 1d downtrend AND volume spike
            elif wr > -20 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (exit oversold) OR 1d trend turns down
            if wr > -50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -50 (exit overbought) OR 1d trend turns up
            if wr < -50 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals