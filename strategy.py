#!/usr/bin/env python3
"""
6h_keltner_1w_trend_volume_v1
Hypothesis: On 6-hour timeframe, use Keltner Channel breakouts with weekly trend filter and volume confirmation. Enter long on upper band breakout in weekly uptrend with volume > 1.5x average, short on lower band breakdown in weekly downtrend with volume > 1.5x average. Exit on opposite band touch. Designed for low frequency (12-37 trades/year) to avoid fee drift while capturing trend continuation. Uses weekly trend filter for better alignment with 6h timeframe. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) by using weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    w_close = df_1w['close'].values
    w_ema50 = pd.Series(w_close).ewm(span=50, adjust=False).mean().values
    w_ema50_aligned = align_htf_to_ltf(prices, df_1w, w_ema50)
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for Keltner Channel (10-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Keltner Channel: EMA20 ± 2*ATR
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA20 warmup
        # Skip if weekly EMA50 not available
        if np.isnan(w_ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs weekly EMA50
        uptrend = close[i] > w_ema50_aligned[i]
        downtrend = close[i] < w_ema50_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below lower Keltner Channel
            if close[i] <= kc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above upper Keltner Channel
            if close[i] >= kc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Keltner Channel in weekly uptrend with volume confirmation
            long_entry = (close[i] > kc_upper[i]) and uptrend and vol_confirm
            # Short entry: price breaks below lower Keltner Channel in weekly downtrend with volume confirmation
            short_entry = (close[i] < kc_lower[i]) and downtrend and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals