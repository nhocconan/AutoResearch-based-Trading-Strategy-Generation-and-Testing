#!/usr/bin/env python3
"""
1d_keltner_channel_1w_trend_volume_v1
Hypothesis: Keltner Channel (20, ATR 2) from daily timeframe acts as dynamic support/resistance.
Price reversals at these levels with volume confirmation and trend alignment capture mean reversion moves.
Works in both bull and bear markets by trading in direction of 1w EMA50 trend.
Targets 7-25 trades/year with disciplined entries to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_channel_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w OHLC for Keltner Channel calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(20) for 1w
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Keltner Channel: EMA(20) ± 2*ATR(20)
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    keltner_upper = ema20_1w + 2 * atr_20
    keltner_lower = ema20_1w - 2 * atr_20
    
    # Align Keltner Channel to 1d timeframe
    keltner_upper_1d = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_1d = align_htf_to_ltf(prices, df_1w, keltner_lower)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_1d[i]) or 
            np.isnan(keltner_upper_1d[i]) or 
            np.isnan(keltner_lower_1d[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches upper band OR trend turns down
            if close[i] >= keltner_upper_1d[i] or close[i] < ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches lower band OR trend turns up
            if close[i] <= keltner_lower_1d[i] or close[i] > ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches lower band + volume confirmation + uptrend
            if (close[i] <= keltner_lower_1d[i] and 
                vol_confirm and 
                close[i] > ema50_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches upper band + volume confirmation + downtrend
            elif (close[i] >= keltner_upper_1d[i] and 
                  vol_confirm and 
                  close[i] < ema50_1d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals