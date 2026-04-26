#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_Volume_Trend_v1
Hypothesis: 4h Keltner Channel (20 EMA ± 2*ATR) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x average volume). Enters long when price breaks above upper Keltner with uptrend and volume, short when breaks below lower Keltner with downtrend and volume. Exits when price retests the 20 EMA (middle line) or reverses across 1d EMA50. Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear by following 1d trend, confirmed by volume to avoid false breakouts. Keltner adapts to volatility via ATR, effective in ranging and trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for EMA, ATR, volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h EMA20 (middle of Keltner)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(10) for Keltner width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel: 20 EMA ± 2*ATR
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50 for 1d EMA, 20 for EMA20, 10 for ATR, 20 for volume)
    start_idx = max(50, 20, 10, 20)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        ema_val = ema_20[i]
        up_val = keltner_upper[i]
        low_val = keltner_lower[i]
        ema_1d_val = ema_50_1d_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(up_val) or np.isnan(low_val) or 
            np.isnan(ema_1d_val) or np.isnan(avg_vol) or np.isnan(atr_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above upper Keltner with 1d uptrend and volume confirmation
        long_condition = (close_val > up_val) and (close_val > ema_1d_val) and volume_confirmed
        # Short logic: price breaks below lower Keltner with 1d downtrend and volume confirmation
        short_condition = (close_val < low_val) and (close_val < ema_1d_val) and volume_confirmed
        
        # Exit logic: 
        # Long exit: price retests or breaks below 20 EMA (middle line) OR closes below 1d EMA50 (trend change)
        long_exit = (position == 1 and (close_val <= ema_val or close_val < ema_1d_val))
        # Short exit: price retests or breaks above 20 EMA (middle line) OR closes above 1d EMA50 (trend change)
        short_exit = (position == -1 and (close_val >= ema_val or close_val > ema_1d_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Keltner_Channel_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0