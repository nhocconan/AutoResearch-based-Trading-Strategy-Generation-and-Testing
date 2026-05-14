#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above H3 in 12h uptrend (price > EMA50).
# Short when price breaks below L3 in 12h downtrend (price < EMA50).
# Volume must be > 2.0x 20-period MA to confirm breakout strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
# This strategy focuses on BTC and ETH as primary targets, using 12h trend filter to avoid counter-trend trades.
# The Camarilla H3/L3 levels provide strong support/resistance with higher breakout validity than H4/L4.
# Volume confirmation ensures breakout validity, and the 12h EMA50 ensures we only trade with the higher timeframe trend.

name = "6h_Camarilla_H3L3_12hEMA50_VolumeSpike"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Camarilla levels from 12h OHLC (using previous bar's close)
    close_12h_shifted = np.roll(close_12h, 1)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    range_12h = high_12h - low_12h
    h3 = close_12h_shifted + range_12h * 1.1 / 4
    l3 = close_12h_shifted - range_12h * 1.1 / 4
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    
    # Volume confirmation: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above H3 AND 12h uptrend AND volume spike
            if close_val > h3_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND 12h downtrend AND volume spike
            elif close_val < l3_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR 12h trend turns down
            if close_val < l3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR 12h trend turns up
            if close_val > h3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals