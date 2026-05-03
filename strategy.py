#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter (EMA50) and volume confirmation.
# Long when Alligator jaws (13-period SMA) > teeth (8-period SMA) in 1w uptrend (price > EMA50).
# Short when Alligator jaws < teeth in 1w downtrend (price < EMA50).
# Volume must be > 1.5x 20-period MA to confirm breakout strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 30-100 total trades over 4 years.

name = "1d_WilliamsAlligator_1wEMA50_Volume"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d: jaws (13-period SMA of median price), teeth (8-period SMA of median price)
    median_price = (high + low) / 2
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaws[i]) or 
            np.isnan(teeth[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: Alligator jaws > teeth AND 1w uptrend AND volume spike
            if jaws[i] > teeth[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Alligator jaws < teeth AND 1w downtrend AND volume spike
            elif jaws[i] < teeth[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator jaws <= teeth OR 1w trend turns down
            if jaws[i] <= teeth[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator jaws >= teeth OR 1w trend turns up
            if jaws[i] >= teeth[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals