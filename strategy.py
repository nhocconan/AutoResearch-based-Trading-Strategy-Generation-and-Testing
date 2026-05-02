#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Spike
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Williams %R(14) identifies overbought/oversold conditions: long when %R crosses above -80 from below,
# short when %R crosses below -20 from above. Works in ranging markets via mean reversion and
# in trending markets via pullbacks to the 1d EMA34. Volume spike (2x 20-period average) confirms
# institutional participation at turning points. Discrete position sizing: 0.25 balances exposure and risk.

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R(14) using prior completed 6h bar's data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().shift(1).values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 1d EMA34 trend (prior completed 1d bar's EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 6h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(14, 34, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA34 (bullish bias) AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA34 (bearish bias) AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -20 (overbought) OR price falls below 1d EMA34 (trend change)
            if williams_r[i] >= -20 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -80 (oversold) OR price rises above 1d EMA34 (trend change)
            if williams_r[i] <= -80 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals