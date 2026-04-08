#!/usr/bin/env python3
# 4h_volume_price_action_v4
# Hypothesis: Price action at 12h high/low levels with volume confirmation and EMA trend filter.
# Trades mean reversion at key levels during high volume, works in both bull/bear by trading reversals.
# Target: 20-30 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_price_action_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h high/low levels for mean reversion
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Use previous 12h high/low (shifted by 1 to avoid look-ahead)
    high_12h_prev = np.concatenate([[np.nan], high_12h[:-1]])
    low_12h_prev = np.concatenate([[np.nan], low_12h[:-1]])
    
    # Align to 4h timeframe
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h_prev)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h_prev)
    
    # EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema50[i]) or np.isnan(high_12h_aligned[i]) or np.isnan(low_12h_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price reaches 12h low or trend turns bearish
            if close[i] <= low_12h_aligned[i] or close[i] < ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches 12h high or trend turns bullish
            if close[i] >= high_12h_aligned[i] or close[i] > ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long setup: price near 12h low in uptrend
                if close[i] <= low_12h_aligned[i] * 1.005 and close[i] > ema50[i]:
                    position = 1
                    signals[i] = 0.25
                # Short setup: price near 12h high in downtrend
                elif close[i] >= high_12h_aligned[i] * 0.995 and close[i] < ema50[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals