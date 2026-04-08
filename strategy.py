#!/usr/bin/env python3
# 4h_12h_momentum_reversal_v1
# Hypothesis: Combines 12h RSI momentum with 4h price action reversal at key levels. Uses 12h RSI to identify overbought/oversold conditions and 4h candlestick patterns for entry timing. Designed for lower trade frequency (<30/year) with clear reversal signals that work in both bull and bear markets by fading extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_momentum_reversal_v1"
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
    
    # Get 12h data for RSI momentum
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 12h close
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    # Align to 4h timeframe (no additional delay needed for RSI)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 4h support/resistance levels from recent swing points
    # Use 20-period high/low for context
    high_4h = df_12h['high'].values if len(df_12h) >= len(df_12h) else df_12h['high'].values  # fallback
    # Actually get proper 4h data for price levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    res_level = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    supp_level = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    res_level_aligned = align_htf_to_ltf(prices, df_4h, res_level)
    supp_level_aligned = align_htf_to_ltf(prices, df_4h, supp_level)
    
    # Volume filter: above average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma * 1.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi_12h_aligned[i]) or np.isnan(res_level_aligned[i]) or np.isnan(supp_level_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI becomes overbought OR price hits resistance
            if rsi_12h_aligned[i] > 70 or close[i] >= res_level_aligned[i] * 0.995:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI becomes oversold OR price hits support
            if rsi_12h_aligned[i] < 30 or close[i] <= supp_level_aligned[i] * 1.005:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI oversold (<30) AND price near support with volume
            if (rsi_12h_aligned[i] < 30 and 
                close[i] <= supp_level_aligned[i] * 1.01 and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought (>70) AND price near resistance with volume
            elif (rsi_12h_aligned[i] > 70 and 
                  close[i] >= res_level_aligned[i] * 0.99 and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals