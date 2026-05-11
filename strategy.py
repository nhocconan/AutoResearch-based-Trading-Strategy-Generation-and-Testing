#!/usr/bin/env python3
name = "4h_RSI_Extremes_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Load 1d data ONCE
    df_1d = get_htf_data(prices, '1d')
    
    # 2. 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 3. 14-period RSI on 4h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4. Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # 5. Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema200 = close[i] > ema200_1d_aligned[i]
        price_below_ema200 = close[i] < ema200_1d_aligned[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: RSI oversold + above 1d EMA200 + volume spike
            if rsi_oversold and price_above_ema200 and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: RSI overbought + below 1d EMA200 + volume spike
            elif rsi_overbought and price_below_ema200 and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: RSI crosses above 50 OR trend reverses
                if rsi[i] > 50 or close[i] < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: RSI crosses below 50 OR trend reverses
                if rsi[i] < 50 or close[i] > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals