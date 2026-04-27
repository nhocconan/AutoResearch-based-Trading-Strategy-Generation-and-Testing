#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_TrendFilter_v1
Hypothesis: Donchian(20) breakouts with volume confirmation and 1d EMA50 trend filter capture strong momentum moves while avoiding false breakouts in ranging markets. 
Discrete sizing (0.30) and ATR-based stoploss (2.0) control risk. Target: 100-180 total trades over 4 years (25-45/year).
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian(20) channels from prior 20 periods (lookback, not including current)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # ATR(14) for dynamic stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d EMA50 to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.30   # Position size: 30% of capital
    
    # Warmup: need Donchian (20), EMA50 (50), volume avg (20), ATR (14)
    start_idx = max(20, 50, 20, 14) + 1  # +1 for shift in Donchian
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Determine trend: price vs 1d EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf:
                # Long breakout: price above upper Donchian band
                if close_val > upper:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short breakdown: price below lower Donchian band
                if close_val < lower:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Long exit: stoploss (2.0*ATR) or price re-enters Donchian channel
            stop_loss = entry_price - 2.0 * atr_val
            if close_val <= stop_loss or close_val < upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: stoploss (2.0*ATR) or price re-enters Donchian channel
            stop_loss = entry_price + 2.0 * atr_val
            if close_val >= stop_loss or close_val > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_Volume_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0