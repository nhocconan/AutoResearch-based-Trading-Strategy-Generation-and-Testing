#!/usr/bin/env python3
"""
6h Williams %R + 1d EMA Trend + Volume Spike Confirmation
- Primary: 6h timeframe for execution
- HTF: 1d for EMA trend filter, 1d for volume spike (using 6h volume vs 6h MA)
- Entry Logic:
  * Williams %R(14) on 6h: oversold < -80 for long, overbought > -20 for short
  * Trend filter: 6h close > 1d EMA(34) for long bias, < for short bias
  * Volume confirmation: current 6h volume > 1.8 * 20-period 6h volume MA
- Exit: Opposite Williams %R signal or volume spike fails
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
- Why should work: Williams %R captures mean reversion in ranges, EMA filter ensures 
  we trade with higher timeframe trend, volume confirmation avoids false signals.
  Works in both bull/bear markets by adapting to regime via EMA trend.
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
    
    # Calculate Williams %R on 6h (primary timeframe)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 6h volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume_spike = volume_spike[i]
        wr = williams_r[i]
        ema_trend = ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if curr_volume_spike:
                # Long conditions: Williams %R oversold AND price above 1d EMA (bullish bias)
                if wr < -80 and curr_close > ema_trend:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: Williams %R overbought AND price below 1d EMA (bearish bias)
                elif wr > -20 and curr_close < ema_trend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R becomes overbought OR volume spike fails OR price crosses below EMA
            if wr > -20 or not curr_volume_spike or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R becomes oversold OR volume spike fails OR price crosses above EMA
            if wr < -80 or not curr_volume_spike or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0