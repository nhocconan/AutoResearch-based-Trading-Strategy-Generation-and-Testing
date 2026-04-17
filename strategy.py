#!/usr/bin/env python3
"""
12h Daily Range Breakout with Volume and Momentum Confirmation
Long: Price breaks above prior day's high + volume > 1.8x 12h volume SMA + RSI(12h) > 50
Short: Price breaks below prior day's low + volume > 1.8x 12h volume SMA + RSI(12h) < 50
Exit: Price crosses opposite daily level
Target: 12-25 trades/year per symbol, tight entry to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get prior 1D high and low (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    prior_1d_high = df_1d['high'].shift(1)  # Prior day's high
    prior_1d_low = df_1d['low'].shift(1)    # Prior day's low
    prior_1d_high_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_high.values)
    prior_1d_low_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_low.values)
    
    # 12h volume SMA for confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_sma_20 = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h = align_htf_to_ltf(prices, df_12h, volume_sma_20)
    
    # 12h RSI for momentum filter (14-period)
    delta = pd.Series(df_12h['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    rsi_12h = align_htf_to_ltf(prices, df_12h, rsi.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1d_high_aligned[i]) or np.isnan(prior_1d_low_aligned[i]) or 
            np.isnan(volume_sma_20_12h[i]) or np.isnan(rsi_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_sma_20_12h[i]
        rsi_val = rsi_12h[i]
        
        if position == 0:
            # Long: break above prior 1D high + volume + momentum
            if price > prior_1d_high_aligned[i] and vol > 1.8 * vol_ma and rsi_val > 50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior 1D low + volume + momentum
            elif price < prior_1d_low_aligned[i] and vol > 1.8 * vol_ma and rsi_val < 50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below prior 1D low
            if price < prior_1d_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior 1D high
            if price > prior_1d_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyRange_Breakout_Volume_Momentum"
timeframe = "12h"
leverage = 1.0