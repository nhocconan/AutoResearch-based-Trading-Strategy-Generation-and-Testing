#!/usr/bin/env python3
"""
1d_WT_1w_Trend_Momentum
Hypothesis: Uses 1d Williams %R momentum combined with 1w EMA trend filter for directional entries.
Williams %R identifies oversold/overbought conditions while 1w EMA provides the higher timeframe trend.
Only takes trades in direction of weekly trend to avoid counter-trend whipsaws. Designed for low
trade frequency by requiring both momentum extreme and trend alignment. Works in bull markets
by buying dips in uptrend and in bear markets by selling rallies in downtrend.
"""

name = "1d_WT_1w_Trend_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Williams %R (14-period) on 1d ---
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # --- 1w EMA50 Trend Filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(williams_r[i]) or np.isnan(ema_50_1d[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Williams %R thresholds: oversold < -80, overbought > -20
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        if position == 0:
            # Long: Williams %R oversold AND price above weekly EMA50 (uptrend)
            if oversold and close[i] > ema_50_1d[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND price below weekly EMA50 (downtrend)
            elif overbought and close[i] < ema_50_1d[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral zone or trend reversal
            if position == 1:
                # Exit long: Williams %R rises above -50 (momentum fading) OR trend breaks
                if williams_r[i] > -50 or close[i] < ema_50_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Williams %R falls below -50 (momentum fading) OR trend breaks
                if williams_r[i] < -50 or close[i] > ema_50_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals