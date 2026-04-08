#!/usr/bin/env python3
# 4h_triple_reversal_volume_v1
# Hypothesis: 4h mean reversion at extreme RSI with volume confirmation and 1d trend filter.
# Long when RSI < 20 and volume > 1.5x average, short when RSI > 80 and volume > 1.5x average.
# Uses 1d EMA50 to filter trades in direction of higher timeframe trend.
# Designed for 20-40 trades/year on 4h to minimize fee drag. Works in ranging markets via mean reversion.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_triple_reversal_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure RSI and volume average are ready
    
    for i in range(start_idx, n):
        # Skip if required data is not available
        if np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) or price closes below EMA50
            if rsi[i] >= 40 or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) or price closes above EMA50
            if rsi[i] <= 60 or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            volume_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long entry: RSI oversold (<20) with volume confirmation and 1d uptrend
            if (rsi[i] < 20 and 
                volume_confirm and 
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought (>80) with volume confirmation and 1d downtrend
            elif (rsi[i] > 80 and 
                  volume_confirm and 
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals