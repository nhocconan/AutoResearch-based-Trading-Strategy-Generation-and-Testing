#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend_1wPivotDir_VolumeConfirm
Hypothesis: Uses 6h timeframe with Camarilla R4/S4 breakouts (strong momentum) filtered by 1d EMA50 trend direction and weekly pivot bias. Only takes breakouts aligned with both the 1d trend and weekly pivot direction to avoid counter-trend trades. Volume confirmation ensures momentum validity. Designed for BTC/ETH to work in both bull and bear markets by requiring trend alignment. Target 12-30 trades/year to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Camarilla levels from previous completed 1d bar (using R4/S4 for strong breakouts)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r4 = prev_close + (rng * 1.1 / 2)  # R4 level
    s4 = prev_close - (rng * 1.1 / 2)  # S4 level
    
    # Align Camarilla R4/S4 levels to 6h
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 1w data for weekly pivot direction (HTF bias)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot points from previous completed 1w bar
    whigh = df_1w['high'].shift(1).values
    wlow = df_1w['low'].shift(1).values
    wclose = df_1w['close'].shift(1).values
    
    # Weekly pivot point and support/resistance levels
    wpivot = (whigh + wlow + wclose) / 3.0
    wr1 = 2 * wpivot - wlow
    ws1 = 2 * wpivot - whigh
    wr2 = wpivot + (whigh - wlow)
    ws2 = wpivot - (whigh - wlow)
    
    # Weekly bias: price above weekly pivot = bullish bias, below = bearish bias
    weekly_bias_bullish = wclose > wpivot
    weekly_bias_bearish = wclose < wpivot
    
    # Align weekly bias to 6h
    weekly_bias_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bullish.astype(float))
    weekly_bias_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bearish.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-period average (moderate threshold)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to minimize fee churn
    
    # Warmup: need 1d EMA50 (50), 1d shift(1) for Camarilla, 1w shift(1) for pivot, vol avg (20)
    start_idx = max(50 + 2*6, 1 + 2*6, 1 + 2*6, 20)  # ~112 bars for 1d EMA50 warmup (1d bars per day = 4)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(weekly_bias_bullish_aligned[i]) or np.isnan(weekly_bias_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        weekly_bull = weekly_bias_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bias_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Look for entry: Camarilla R4/S4 breakout with 1d EMA50 alignment, weekly pivot bias, and volume confirmation
            long_condition = (close_val > r4_val and 
                            close_val > ema_val and 
                            weekly_bull and 
                            vol_conf)
            short_condition = (close_val < s4_val and 
                             close_val < ema_val and 
                             weekly_bear and 
                             vol_conf)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_1wPivotDir_VolumeConfirm"
timeframe = "6h"
leverage = 1.0