#!/usr/bin/env python3
# 4h_ema_trend_volume_breakout_v1
# Hypothesis: EMA trend alignment (4h EMA50 > 200) with volume breakout (volume > 2x 20-period avg) captures strong trends in both bull and bear markets.
# Uses 12h EMA200 as higher timeframe filter to avoid counter-trend trades. Position size 0.25 for capital preservation.
# Target: 20-30 trades/year (80-120 over 4 years) with focus on quality over quantity.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_trend_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA calculations
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 2.0
    
    # Higher timeframe trend filter (12h EMA200)
    df_12h = get_htf_data(prices, '12h')
    ema_200_12h = pd.Series(df_12h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Wait for EMA200 warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend based on 4h EMA alignment
        uptrend_4h = ema_50[i] > ema_200[i]
        downtrend_4h = ema_50[i] < ema_200[i]
        
        # Higher timeframe filter: only trade in direction of 12h trend
        uptrend_12h = close[i] > ema_200_12h_aligned[i]
        downtrend_12h = close[i] < ema_200_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: trend reversal or volume drops below average
            if not uptrend_4h or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend reversal or volume drops below average
            if not downtrend_4h or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: 4h uptrend + 12h uptrend + volume breakout
            if (uptrend_4h and uptrend_12h and volume[i] > vol_threshold[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: 4h downtrend + 12h downtrend + volume breakout
            elif (downtrend_4h and downtrend_12h and volume[i] > vol_threshold[i]):
                position = -1
                signals[i] = -0.25
    
    return signals