#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d trend filter (EMA50) and volume confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) in 1d uptrend (close > EMA50).
# Short when Bull Power < 0 AND Bear Power > 0 (bearish momentum) in 1d downtrend (close < EMA50).
# Volume must be > 1.3x 20-period MA to confirm strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "6h_ElderRay_1dEMA50_Volume"
timeframe = "6h"
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
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: current volume > 1.3x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d uptrend AND volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND 1d downtrend AND volume spike
            elif bull_power[i] < 0 and bear_power[i] > 0 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum turns bearish OR 1d trend turns down
            if bull_power[i] < 0 or bear_power[i] > 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum turns bullish OR 1d trend turns up
            if bull_power[i] > 0 or bear_power[i] < 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals