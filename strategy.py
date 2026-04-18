#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_Volume
1d strategy using weekly Keltner Channel with volume confirmation and weekly trend filter.
- Long: Close breaks above weekly upper Keltner + volume > 1.5x weekly avg + weekly EMA21 > EMA50
- Short: Close breaks below weekly lower Keltner + volume > 1.5x weekly avg + weekly EMA21 < EMA50
- Exit: Opposite breakout or trend reversal
Designed for ~10-25 trades/year per symbol (40-100 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Keltner Channel and filters
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Keltner Channel for weekly timeframe
    # EMA21 of close
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    # ATR(10) of weekly data
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])  # align length
    atr_10_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Keltner Channel
    upper_keltner = ema_21_1w + 2 * atr_10_1w
    lower_keltner = ema_21_1w - 2 * atr_10_1w
    
    # Weekly EMA21 and EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly volume average (20-period)
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all weekly data to daily timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_21_aligned[i] > ema_50_aligned[i]
        downtrend = ema_21_aligned[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > upper_keltner_aligned[i]
        breakdown_down = close[i] < lower_keltner_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above weekly upper Keltner
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below weekly lower Keltner
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below weekly lower Keltner
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above weekly upper Keltner
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Channel_Breakout_Volume"
timeframe = "1d"
leverage = 1.0