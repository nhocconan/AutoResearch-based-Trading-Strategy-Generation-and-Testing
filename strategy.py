#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h RSI and 1d MACD for trend direction with 1h volume spike for entry timing.
- 4h RSI(14) > 50 = bullish bias, < 50 = bearish bias
- 1d MACD histogram > 0 = bullish bias, < 0 = bearish bias
- Enter long when both 4h RSI > 50 and 1d MACD hist > 0 AND 1h volume > 2.0 x 20-period volume MA
- Enter short when both 4h RSI < 50 and 1d MACD hist < 0 AND 1h volume > 2.0 x 20-period volume MA
- Exit when either 4h RSI crosses 50 or 1d MACD hist crosses zero
- Fixed position size 0.20 to manage drawdown and limit trades
- Uses multi-timeframe alignment to avoid look-ahead
- Target: 15-37 trades/year (60-150 over 4 years) for 1h timeframe
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
    
    # Get 4h data for RSI filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    delta = pd.Series(df_4h['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_values = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_values)
    
    # Get 1d data for MACD filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate 1d MACD (12,26,9)
    close_1d = pd.Series(df_1d['close'])
    ema_fast = close_1d.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close_1d.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - signal_line
    macd_hist_values = macd_hist.values
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist_values)
    
    # Volume confirmation: 20-period volume MA on 1h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(rsi_4h_aligned[i]) or np.isnan(macd_hist_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        rsi = rsi_4h_aligned[i]
        macd_hist = macd_hist_aligned[i]
        
        if position == 0:
            # Look for aligned signals with volume spike
            # Long: 4h RSI > 50, 1d MACD hist > 0, volume spike
            if rsi > 50 and macd_hist > 0 and vol > 2.0 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short: 4h RSI < 50, 1d MACD hist < 0, volume spike
            elif rsi < 50 and macd_hist < 0 and vol > 2.0 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit when either 4h RSI < 50 or 1d MACD hist < 0
            if rsi < 50 or macd_hist < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit when either 4h RSI > 50 or 1d MACD hist > 0
            if rsi > 50 or macd_hist > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI4h_MACD1d_VolumeSpike"
timeframe = "1h"
leverage = 1.0