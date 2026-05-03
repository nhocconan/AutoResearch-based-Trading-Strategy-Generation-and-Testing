#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) in 1d uptrend (price > EMA50).
# Short when Williams %R > -20 (overbought) in 1d downtrend (price < EMA50).
# Volume must be > 1.5x 20-period MA to confirm reversal strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.
# Williams %R is effective in ranging and bear markets, capturing reversals at extremes.
# The 1d EMA50 ensures we trade with the higher timeframe trend, avoiding counter-trend trades.
# Volume confirmation adds validity to the reversal signal.

name = "4h_WilliamsR_1dEMA50_VolumeConfirm"
timeframe = "4h"
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
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R on 4h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        williams_r_val = williams_r[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        vol_conf = volume_confirm[i]
        
        # Entry logic
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND 1d uptrend AND volume confirmation
            if williams_r_val < -80 and trend_up and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND 1d downtrend AND volume confirmation
            elif williams_r_val > -20 and trend_down and vol_conf:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (exit oversold) OR 1d trend turns down
            if williams_r_val > -50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -50 (exit overbought) OR 1d trend turns up
            if williams_r_val < -50 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals