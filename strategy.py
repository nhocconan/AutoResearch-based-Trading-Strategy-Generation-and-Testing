#!/usr/bin/env python3
"""
1d_1w_Keltner_Breakout_Trend_Filter_v1
Hypothesis: On 1d timeframe, take long when price breaks above Keltner upper band with bullish 1w EMA trend, and short when price breaks below Keltner lower band with bearish 1w EMA trend. Uses 1w EMA trend filter to avoid counter-trend breakouts, reducing whipsaws. Designed for 15-25 trades/year by requiring strong trend alignment and breakout confirmation. Works in bull markets via upper band breaks and bear markets via lower band breaks, while avoiding ranging markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Keltner_Breakout_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Keltner Channel (20, 10) on 1d
    atr_period = 20
    ma_period = 20
    multiplier = 2.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Middle line (EMA)
    ema_middle = pd.Series(close).ewm(span=ma_period, adjust=False, min_periods=ma_period).mean().values
    
    # Keltner bands
    keltner_upper = ema_middle + multiplier * atr
    keltner_lower = ema_middle - multiplier * atr
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA50
        bullish_trend = close[i] > ema_50_1w_aligned[i]
        bearish_trend = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > keltner_upper[i]
        breakout_down = close[i] < keltner_lower[i]
        
        # Entry conditions: breakout + trend alignment
        long_entry = breakout_up and bullish_trend
        short_entry = breakout_down and bearish_trend
        
        # Exit conditions: price returns to middle line (mean reversion)
        long_exit = close[i] < ema_middle[i]
        short_exit = close[i] > ema_middle[i]
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals