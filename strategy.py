#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend and Williams %R calculation.
- Williams %R(14) identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold.
- Trend filter: 1d EMA(50) - price above EMA = bullish bias, below = bearish bias.
- Entry logic: In bullish trend (price > 1d EMA50): long when Williams %R crosses above -80 from below (oversold bounce).
               In bearish trend (price < 1d EMA50): short when Williams %R crosses below -20 from above (overbought rejection).
- Volume confirmation: current 6h volume > 1.5 * 20-period volume MA to avoid false signals.
- Exit: Opposite Williams %R crossover (long exits when %R crosses below -50, short exits when %R crosses above -50) or trend reversal.
- Discrete signal size: 0.25 to manage drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 1d
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_1d - df_1d['close'].values) / (highest_high_1d - lowest_low_1d + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Volume confirmation on 6h: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # EMA50, Williams %R14, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50 = ema_50_aligned[i]
        williams_r = williams_r_aligned[i]
        prev_williams_r = williams_r_aligned[i-1] if i > 0 else williams_r
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish trend: price above 1d EMA50
                if curr_close > ema_50:
                    # Long entry: Williams %R crosses above -80 from below (oversold bounce)
                    if prev_williams_r <= -80 and williams_r > -80:
                        signals[i] = 0.25
                        position = 1
                # Bearish trend: price below 1d EMA50
                elif curr_close < ema_50:
                    # Short entry: Williams %R crosses below -20 from above (overbought rejection)
                    if prev_williams_r >= -20 and williams_r < -20:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (loss of momentum) OR trend reversal
            if williams_r < -50 or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (loss of momentum) OR trend reversal
            if williams_r > -50 or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0