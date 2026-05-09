#!/usr/bin/env python3
# 6h_LiquiditySweep_Reversal_12hTrend
# Hypothesis: On 6h timeframe, liquidity sweeps (price briefly breaks swing high/low then reverses)
# followed by 12h trend continuation capture high-probability reversals in both bull/bear markets.
# Uses 12h trend filter to avoid counter-trend trades. Entry on reversal confirmation with volume spike.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).

name = "6h_LiquiditySweep_Reversal_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h trend: EMA(34) on close
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_12h = close_12h > ema_34_12h
    
    # Swing points on 6h: lookback 20 periods
    lookback = 20
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        swing_high[i] = np.max(high[i-lookback:i])
        swing_low[i] = np.min(low[i-lookback:i])
    
    # Align 12h trend to 6h
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = lookback + 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trend_up_12h_aligned[i]) or np.isnan(swing_high[i]) or np.isnan(swing_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish liquidity sweep reversal: price breaks swing low then closes above it
            if (low[i] < swing_low[i] and 
                close[i] > swing_low[i] and 
                trend_up_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Bearish liquidity sweep reversal: price breaks swing high then closes below it
            elif (high[i] > swing_high[i] and 
                  close[i] < swing_high[i] and 
                  not trend_up_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks swing low or trend changes
            if low[i] < swing_low[i] or not trend_up_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks swing high or trend changes
            if high[i] > swing_high[i] or trend_up_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals