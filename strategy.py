#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h, HTF: 1d for trend filter
- Williams %R(14) measures overbought/oversold levels (-20 = overbought, -80 = oversold)
- Long: Williams %R crosses above -80 from below + price > 1d EMA34 (uptrend) + volume > 1.8x 20-period avg
- Short: Williams %R crosses below -20 from above + price < 1d EMA34 (downtrend) + volume > 1.8x 20-period avg
- Exit: Williams %R returns to -50 (mean reversion) OR opposite extreme reached
- Uses tighter volume confirmation (1.8x) to reduce trades and avoid fee drag
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to balance return and risk
- BTC/ETH focus: requires HTF trend alignment to avoid SOL-only bias
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
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
    
    # Volume confirmation: > 1.8x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where -20 is overbought, -80 is oversold
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # Need 20 for volume MA, 14 for Williams %R, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        # Williams %R levels
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below (oversold bounce) + uptrend + volume spike
            if (wr > -80 and wr_prev <= -80 and  # Cross above -80
                close[i] > ema_34_aligned[i] and  # Uptrend filter
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (overbought rejection) + downtrend + volume spike
            elif (wr < -20 and wr_prev >= -20 and  # Cross below -20
                  close[i] < ema_34_aligned[i] and  # Downtrend filter
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) OR reaches overbought (-20)
            if wr >= -50 or wr >= -20:  # Return to mean or reach overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) OR reaches oversold (-80)
            if wr <= -50 or wr <= -80:  # Return to mean or reach oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0