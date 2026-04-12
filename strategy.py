#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Weekly ATR breakout: Long when price breaks above weekly ATR-based ceiling,
# Short when breaks below floor. Uses 1d price action with 1w volatility filter.
# Works in bull (breakouts) and bear (mean-reversion from extremes).
name = "1d_1w_atr_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for ATR calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(14) on weekly data
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]  # first bar
    tr2[0] = high_1w[0] - close_1w[0]
    tr3[0] = low_1w[0] - close_1w[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly midpoint
    mid_1w = (high_1w + low_1w) / 2.0
    
    # ATR-based bands: midpoint ± 1.5 * ATR
    upper_1w = mid_1w + 1.5 * atr_1w
    lower_1w = mid_1w - 1.5 * atr_1w
    
    # Align weekly bands to daily timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if bands not ready
        if np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout signals
        long_breakout = close[i] > upper_1w_aligned[i]
        short_breakout = close[i] < lower_1w_aligned[i]
        
        # Exit when price returns to weekly midpoint
        reenter_long = close[i] < mid_1w[i] if not np.isnan(mid_1w[i]) else False
        reenter_short = close[i] > mid_1w[i] if not np.isnan(mid_1w[i]) else False
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif reenter_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif reenter_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals