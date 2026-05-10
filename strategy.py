# 4h_Bollinger_Band_Width_Squeeze_Breakout
# Hypothesis: Bollinger Band Width contraction (low volatility) precedes breakout moves.
# We enter long when BB Width is at 20-period low AND price breaks above upper band.
# We enter short when BB Width is at 20-period low AND price breaks below lower band.
# This captures volatility breakouts after consolidation, effective in both bull and bear markets.
# Uses volume confirmation to avoid false breakouts and ATR for dynamic position sizing.
# Timeframe: 4h, with 1d trend filter to align with higher timeframe momentum.

name = "4h_Bollinger_Band_Width_Squeeze_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (20-period lookback)
    bb_width_series = bb_width.values
    bb_width_rank = pd.Series(bb_width_series).rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need BB calculations (20), EMA50_1d (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bb_width_rank[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB Width at 20-period low (bottom 20% of rank)
        squeeze_condition = bb_width_rank[i] <= 0.2
        
        # Breakout conditions
        breakout_up = close[i] > bb_upper[i]
        breakout_down = close[i] < bb_lower[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: squeeze + upward breakout + volume + uptrend
            if squeeze_condition and breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze + downward breakout + volume + downtrend
            elif squeeze_condition and breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below middle band or trend reversal
            if close[i] < bb_middle[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above middle band or trend reversal
            if close[i] > bb_middle[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3