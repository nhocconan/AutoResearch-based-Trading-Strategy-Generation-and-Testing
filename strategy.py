#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily price breaks above/below Keltner Channel (20, 2.0) with weekly EMA50 trend filter and volume confirmation (>1.5x 20-day average volume). 
Keltner Channels adapt to volatility via ATR, providing dynamic support/resistance that works in both trending and ranging markets. 
Weekly trend filter ensures trades align with higher-timeframe momentum, reducing false breakouts in choppy conditions. 
Volume confirmation adds conviction to breakouts. Designed for low trade frequency (~10-20/year) to minimize fee drag and work across BTC/ETH/SOL.
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
    
    # Get 1d data for indicators (already daily, but using helper for consistency)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ATR(20) for Keltner Channel
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate EMA(20) of close for Keltner Channel middle line
    ema20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bounds
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR/EMA20 (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with trend filter and volume confirmation
            # Long: price breaks above upper KC in uptrend (close > EMA50_1w) with volume confirmation
            # Short: price breaks below lower KC in downtrend (close < EMA50_1w) with volume confirmation
            long_signal = (close[i] > upper[i]) and (close[i] > ema50_1w_aligned[i]) and volume_confirmed[i]
            short_signal = (close[i] < lower[i]) and (close[i] < ema50_1w_aligned[i]) and volume_confirmed[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below middle line (EMA20) or weekly trend turns bearish
            exit_signal = (close[i] < ema20[i]) or (close[i] < ema50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above middle line (EMA20) or weekly trend turns bullish
            exit_signal = (close[i] > ema20[i]) or (close[i] > ema50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Keltner_Channel_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0