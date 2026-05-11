#!/usr/bin/env python3
"""
12h_TRIX_Trend_Filtered_With_Volume
Hypothesis: TRIX momentum combined with 1d trend filter and volume spikes provides reliable entries.
TRIX filters noise and captures sustained momentum, while volume confirms institutional participation.
Trades only in direction of higher timeframe trend to avoid counter-trend whipsaws.
Designed for low frequency (12-25 trades/year) to minimize fee drag in both bull and bear markets.
"""

name = "12h_TRIX_Trend_Filtered_With_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- TRIX (15-period EMA of EMA of EMA of price change) ---
    # Calculate ROC (rate of change)
    roc = np.diff(close, prepend=close[0])
    # Triple EMA
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.values
    
    # --- 1d EMA50 for trend filter ---
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Volume Spike (12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: TRIX momentum with volume and trend alignment
        long_entry = (trix[i] > 0) and vol_spike[i] and (close[i] > ema_50_1d_aligned[i])
        short_entry = (trix[i] < 0) and vol_spike[i] and (close[i] < ema_50_1d_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: TRIX crosses zero or trend reversal
            if position == 1:
                # Exit if TRIX turns negative or trend turns down
                if (trix[i] < 0) or (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit if TRIX turns positive or trend turns up
                if (trix[i] > 0) or (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals