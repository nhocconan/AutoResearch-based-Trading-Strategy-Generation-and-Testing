#!/usr/bin/env python3
"""
1d_RSI_MeanReversion_WeeklyTrend
Hypothesis: In the 1d timeframe, RSI extremes combined with weekly trend alignment provide high-probability mean-reversion entries.
Weekly trend filter (price vs 1w EMA50) avoids counter-trend trades in strong trends, while RSI < 30/ > 70 captures overextended moves.
Volume confirmation ensures institutional participation. Designed for low trade frequency (<25/year) to minimize fee drag.
Works in both bull and bear markets by aligning with weekly trend direction.
"""

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
    
    # Get 1d data for RSI calculation (using close prices)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily closes
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1w data for weekly trend filter (price vs EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average (using 1d volume avg aligned)
    vol_1d = df_1d['volume'].values
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = vol_1d > (1.5 * vol_avg)
    
    # Align all indicators to primary timeframe (1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need RSI (14+14=28), EMA50 (50), volume avg (20)
    start_idx = max(28, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        rsi_val = rsi_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine weekly trend: price vs EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            # Long setup: RSI oversold (<30) in weekly uptrend with volume
            if uptrend and rsi_val < 30 and vol_conf:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short setup: RSI overbought (>70) in weekly downtrend with volume
            elif downtrend and rsi_val > 70 and vol_conf:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: RSI returns to neutral (>50) or weekly trend breaks
            if rsi_val > 50 or close_val < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: RSI returns to neutral (<50) or weekly trend breaks
            if rsi_val < 50 or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI_MeanReversion_WeeklyTrend"
timeframe = "1d"
leverage = 1.0