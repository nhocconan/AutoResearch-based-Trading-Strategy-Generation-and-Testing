#!/usr/bin/env python3
"""
12h_1d_keltner_breakout_volume
Hypothesis: 12h Keltner Channel breakout with 1d volume confirmation.
In bull markets: buy breakouts above upper Keltner (20, 2.0) with volume surge.
In bear markets: sell breakdowns below lower Keltner with volume surge.
Uses 1d volume > 1.5x 20-period average to confirm institutional participation.
Keltner Channels adapt to volatility, reducing false breakouts in low-vol regimes.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 12h Keltner Channel (20, 2.0)
    # Middle = EMA20 of close
    # Upper = Middle + 2.0 * ATR(10)
    # Lower = Middle - 2.0 * ATR(10)
    close_series = pd.Series(close)
    ema_middle = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # True Range for ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    keltner_upper = ema_middle + 2.0 * atr
    keltner_lower = ema_middle - 2.0 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_middle[i]) or 
            np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Keltner breakout + volume spike
        breakout_long = close[i] > keltner_upper[i]
        breakout_short = close[i] < keltner_lower[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        
        long_entry = breakout_long and vol_confirm
        short_entry = breakout_short and vol_confirm
        
        # Exit when price returns to middle line (mean reversion)
        exit_long = position == 1 and close[i] < ema_middle[i]
        exit_short = position == -1 and close[i] > ema_middle[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_keltner_breakout_volume"
timeframe = "12h"
leverage = 1.0