#4h_12h_ema_cross_vol_breakout_v1
# Hypothesis: 4-hour EMA crossover with 12-hour trend filter and volume confirmation.
# Long when fast EMA crosses above slow EMA, price above 12h EMA, and volume > 1.5x average.
# Short when fast EMA crosses below slow EMA, price below 12h EMA, and volume > 1.5x average.
# Exit when EMA cross reverses or volume drops below average.
# Uses EMA on 4h for entry timing and 12h for trend filter to avoid counter-trend trades.
# Designed to generate ~20-30 trades/year to avoid fee decay while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_ema_cross_vol_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA on 4h data
    fast_period = 9
    slow_period = 21
    
    # Fast EMA
    ema_fast = np.full(n, np.nan)
    ema_fast[fast_period-1] = np.mean(close[:fast_period])
    for i in range(fast_period, n):
        ema_fast[i] = (close[i] * 2/(fast_period+1)) + (ema_fast[i-1] * (1-2/(fast_period+1)))
    
    # Slow EMA
    ema_slow = np.full(n, np.nan)
    ema_slow[slow_period-1] = np.mean(close[:slow_period])
    for i in range(slow_period, n):
        ema_slow[i] = (close[i] * 2/(slow_period+1)) + (ema_slow[i-1] * (1-2/(slow_period+1)))
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA on 12-hour data
    ema_12h = np.full(len(close_12h), np.nan)
    ema_12h[20] = np.mean(close_12h[:21])
    for i in range(21, len(close_12h)):
        ema_12h[i] = (close_12h[i] * 2/22) + (ema_12h[i-1] * (1-2/22))
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate average volume for confirmation
    vol_ma = np.full(n, np.nan)
    vol_ma[19] = np.mean(volume[:20])
    for i in range(20, n):
        vol_ma[i] = (volume[i] + vol_ma[i-1] * 19) / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        ema_fast_val = ema_fast[i]
        ema_slow_val = ema_slow[i]
        ema_12h_val = ema_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        volume_val = volume[i]
        
        if position == 1:  # Long
            # Exit: EMA cross reverses or volume drops below average
            if ema_fast_val < ema_slow_val or volume_val < vol_ma_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: EMA cross reverses or volume drops below average
            if ema_fast_val > ema_slow_val or volume_val < vol_ma_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: EMA crossover with trend filter and volume confirmation
            # Bullish: fast EMA crosses above slow EMA, price above 12h EMA, volume > 1.5x average
            if (ema_fast_val > ema_slow_val and 
                ema_fast_val <= ema_slow_val + 1e-9 and  # Ensure crossover just happened
                price > ema_12h_val and 
                volume_val > vol_ma_val * 1.5):
                position = 1
                signals[i] = 0.25
            # Bearish: fast EMA crosses below slow EMA, price below 12h EMA, volume > 1.5x average
            elif (ema_fast_val < ema_slow_val and 
                  ema_fast_val >= ema_slow_val - 1e-9 and  # Ensure crossover just happened
                  price < ema_12h_val and 
                  volume_val > vol_ma_val * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals