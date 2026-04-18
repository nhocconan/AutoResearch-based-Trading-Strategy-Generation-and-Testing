#!/usr/bin/env python3
"""
12h_RSI_Overbought_Oversold_with_Volume_Confirmation
Hypothesis: RSI extremes on 12h chart with volume confirmation and 1d trend filter.
In bull markets, buy when RSI < 30 (oversold) and price > EMA50; in bear markets, sell when RSI > 70 (overbought) and price < EMA50.
Uses volume spike to confirm momentum and avoid false signals. Designed for low trade frequency (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI aligned to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Warmup for RSI, EMA, volume
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_aligned[i]
        ema50 = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: RSI oversold (<30) with volume spike and uptrend (price > EMA50)
            if rsi_val < 30 and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) with volume spike and downtrend (price < EMA50)
            elif rsi_val > 70 and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: RSI > 50 (mean reversion) or trend turns down
            if rsi_val > 50 or price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: RSI < 50 (mean reversion) or trend turns up
            if rsi_val < 50 or price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_RSI_Overbought_Oversold_with_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0