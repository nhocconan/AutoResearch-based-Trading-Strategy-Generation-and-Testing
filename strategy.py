#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation.
- Williams %R(14) identifies overbought/oversold conditions on 12h timeframe.
- Long: Williams %R crosses above -80 (oversold recovery) with volume > 2.0x 20-bar average and price > 1d EMA34.
- Short: Williams %R crosses below -20 (overbought rejection) with volume > 2.0x 20-bar average and price < 1d EMA34.
- Trend filter: price must be above/below 1d EMA34 to align with daily trend.
- Designed for 12h timeframe to capture multi-day reversals with higher probability entries.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false signals in choppy markets.
- Novelty: Combines Williams %R momentum oscillator with 1d EMA trend filter on 12h timeframe for BTC/ETH edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Long: Williams %R crosses above -80 (oversold recovery) AND price > 1d EMA34
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA34
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum loss) OR price < 1d EMA34
            if williams_r[i] < -50 and williams_r[i-1] >= -50 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum loss) OR price > 1d EMA34
            if williams_r[i] > -50 and williams_r[i-1] <= -50 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Reversal_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0