#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R extreme reversal with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h, HTF: 1w for trend filter (avoids SOL bias, works for BTC/ETH)
- Williams %R(14) measures overbought/oversold: > -20 = overbought, < -80 = oversold
- Long: Williams %R crosses above -80 from below + price > 1w EMA50 (uptrend) + volume > 2.0x 20-period avg
- Short: Williams %R crosses below -20 from above + price < 1w EMA50 (downtrend) + volume > 2.0x 20-period avg
- Exit: Williams %R returns to opposite extreme (-80 for long exit, -20 for short exit)
- Uses Williams %R extremes for controlled entries on 12h (mean reversion in trends)
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to balance return and risk
- BTC/ETH focus: requires 1w HTF trend alignment to avoid SOL-only bias
- Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend)
- Uses mtf_data helper for proper HTF alignment without look-ahead
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
    
    # Volume confirmation: > 2.0x 20-period average (strong volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below + price > 1w EMA50 (uptrend) + volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above + price < 1w EMA50 (downtrend) + volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -80 (overbought threshold)
            if williams_r[i] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -20 (oversold threshold)
            if williams_r[i] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0