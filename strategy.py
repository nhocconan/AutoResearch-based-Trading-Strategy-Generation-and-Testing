#!/usr/bin/env python3
# 4h_rsi_volume_momentum_v1
# Hypothesis: On 4h timeframe, combine RSI momentum with volume confirmation and 1d trend filter.
# Long when RSI crosses above 50 with volume > 1.5x average and 1d trend up (price > EMA50).
# Short when RSI crosses below 50 with volume > 1.5x average and 1d trend down (price < EMA50).
# Exit when RSI returns to neutral zone (40-60) or volume drops below average.
# Uses 1d EMA50 for trend filter to avoid whipsaws in ranging markets.
# Designed for ~25-40 trades/year to minimize fee drag while capturing momentum moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_volume_momentum_v1"
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
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) or volume drops below average
            if rsi[i] >= 40 and rsi[i] <= 60 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) or volume drops below average
            if rsi[i] >= 40 and rsi[i] <= 60 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # 1d trend filter
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long entry: RSI crosses above 50 with volume and uptrend
            if rsi[i] > 50 and rsi[i-1] <= 50 and volume_ok and uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI crosses below 50 with volume and downtrend
            elif rsi[i] < 50 and rsi[i-1] >= 50 and volume_ok and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals