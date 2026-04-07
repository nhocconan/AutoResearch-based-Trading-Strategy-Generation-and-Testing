#!/usr/bin/env python3
"""
4h_keltner_pullback_1d_trend_volume_v1
Hypothesis: Keltner Channel pullback strategy on 4h with daily trend filter and volume confirmation.
In trending markets (price above/below daily EMA200), enter long on pullback to lower Keltner band (EMA20 - 2*ATR)
or short on pullback to upper band (EMA20 + 2*ATR). Volume confirmation filters weak signals.
Works in bull markets via trend-following pullsbacks and in bear markets via mean-reversion
within the channel during consolidation phases. Targets 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_keltner_pullback_1d_trend_volume_v1"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    
    # Align daily EMA200 to 4h timeframe
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h indicators: EMA20 and ATR(14) for Keltner Channels
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Keltner Channel bands
    keltner_upper = ema20 + 2 * atr14
    keltner_lower = ema20 - 2 * atr14
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema200_4h[i]) or np.isnan(ema20[i]) or 
            np.isnan(atr14[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA20 (trend change) OR 
            # hits upper Keltner band (mean reversion target)
            if close[i] < ema20[i] or close[i] > keltner_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above EMA20 (trend change) OR
            # hits lower Keltner band (mean reversion target)
            if close[i] > ema20[i] or close[i] < keltner_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Pullback long: price touches lower Keltner band in uptrend
            if (close[i] <= keltner_lower[i] and 
                vol_confirm and 
                close[i] > ema200_4h[i]):
                position = 1
                signals[i] = 0.25
            # Pullback short: price touches upper Keltner band in downtrend
            elif (close[i] >= keltner_upper[i] and 
                  vol_confirm and 
                  close[i] < ema200_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals