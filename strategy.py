#!/usr/bin/env python3
"""
1h_TripleConfirm_TrendBreak_v1
Hypothesis: Combine 4h trend (EMA50), 1d momentum (RSI>50), and 1h volume spike 
with Donchian breakout for high-probability entries. Uses 1h only for timing, 
4h/1d for direction/filter. Target 15-35 trades/year to avoid fee drag. 
Works in bull (trend continuation) and bear (mean reversion within trend).
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
    
    # 1h Donchian breakout (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h trend: EMA50
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d momentum: RSI(14) > 50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # 1h volume confirmation: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need Donchian (20), EMA50 (50), RSI (14), vol avg (20)
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema50 = ema50_4h_aligned[i]
        rsi = rsi_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Determine 4h trend
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            # Long: uptrend + RSI>50 + volume spike + break above Donchian high
            if uptrend and rsi > 50 and vol_spike and close_val > upper:
                signals[i] = size
                position = 1
            # Short: downtrend + RSI<50 + volume spike + break below Donchian low
            elif downtrend and rsi < 50 and vol_spike and close_val < lower:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: trend reversal (close below 4h EMA50) or loss of momentum
            if close_val < ema50 or rsi < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: trend reversal (close above 4h EMA50) or loss of momentum
            if close_val > ema50 or rsi > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_TripleConfirm_TrendBreak_v1"
timeframe = "1h"
leverage = 1.0