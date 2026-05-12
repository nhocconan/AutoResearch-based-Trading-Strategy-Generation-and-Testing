#!/usr/bin/env python3
name = "12h_Donchian_Breakout_1dTrend_VolumeSqueeze"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Data for trend and squeeze ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d EMA34 for trend ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Bollinger Bands for squeeze ===
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + (2.0 * std20_1d)
    lower_bb_1d = sma20_1d - (2.0 * std20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma20_1d
    # Squeeze when BB width is below 20-period percentile (low volatility)
    bb_width_pct_1d = pd.Series(bb_width_1d).rolling(window=20, min_periods=20).rank(pct=True).values
    squeeze_condition = bb_width_pct_1d < 0.2  # Bottom 20% = squeeze
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition.astype(float))
    
    # === 12h Donchian channels (20-period) ===
    # Use rolling window on 12h data directly
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Donchian breakout above + 1d trend up + volatility squeeze
            if (high[i] > highest_high_20[i] and 
                close[i] > ema34_1d_aligned[i] and
                squeeze_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below + 1d trend down + volatility squeeze
            elif (low[i] < lowest_low_20[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  squeeze_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakdown or trend reversal
            if low[i] < lowest_low_20[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout or trend reversal
            if high[i] > highest_high_20[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals