# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_Weekly_Trend_Follower
Hypothesis: Uses weekly trend direction (EMA34) to guide daily entries at support/resistance (weekly ATR-based bands). Low trade frequency (target 10-20/year) to minimize fee drag, works in bull/bear by only trading with weekly trend.
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
    
    # Get weekly data for trend and bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly ATR(14) for dynamic bands
    tr1w = np.maximum(
        df_1w['high'].values - df_1w['low'].values,
        np.maximum(
            np.abs(df_1w['high'].values - np.roll(weekly_close, 1)),
            np.abs(df_1w['low'].values - np.roll(weekly_close, 1))
        )
    )
    tr1w[0] = 0  # First period has no previous close
    atr14_1w = pd.Series(tr1w).rolling(window=14, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, tr1w)
    atr14_1w_aligned = pd.Series(atr14_1w_aligned).rolling(window=14, min_periods=14).mean().values
    
    # Weekly upper/lower bands: EMA34 ± 1.5 * ATR14
    upper_band = ema34_1w + 1.5 * atr14_1w
    lower_band = ema34_1w - 1.5 * atr14_1w
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA and ATR
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema34_val = ema34_1w_aligned[i]
        upper_val = upper_band_aligned[i]
        lower_val = lower_band_aligned[i]
        
        if position == 0:
            # Long: weekly uptrend and price touches lower band (support)
            if close[i] > ema34_val and close[i] <= lower_val:
                signals[i] = size
                position = 1
            # Short: weekly downtrend and price touches upper band (resistance)
            elif close[i] < ema34_val and close[i] >= upper_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses above weekly EMA34 (take profit) or crosses below lower band (stop)
            if close[i] >= ema34_val or close[i] < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses below weekly EMA34 (take profit) or crosses above upper band (stop)
            if close[i] <= ema34_val or close[i] > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Trend_Follower"
timeframe = "1d"
leverage = 1.0