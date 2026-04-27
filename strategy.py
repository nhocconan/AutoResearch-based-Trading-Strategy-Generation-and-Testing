#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeATR_Stop
Hypothesis: 4h Donchian(20) breakout in direction of 12h EMA50 trend with volume confirmation (>1.5x average). ATR-based trailing stop (2.5 ATR) manages risk. Designed for 15-25 trades/year per symbol to avoid fee drag. Works in bull markets via upside breakouts and bear markets via downside breakdowns. Uses discrete position sizing (0.25) to minimize churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility and trailing stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Donchian (20), EMA50 (50), ATR (14), volume avg (20)
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_max_val = high_max[i]
        low_min_val = low_min[i]
        ema_12h_val = ema_50_12h_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price > EMA50 = uptrend, price < EMA50 = downtrend
            is_uptrend = close_val > ema_12h_val
            is_downtrend = close_val < ema_12h_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above Donchian high and volume confirms
                if (close_val > high_max_val) and vol_conf:
                    signals[i] = size
                    position = 1
                    entry_price = close_val  # Track for trailing stop
                    highest_since_entry = close_val
            elif is_downtrend:
                # Downtrend: short when price breaks below Donchian low and volume confirms
                if (close_val < low_min_val) and vol_conf:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val  # Track for trailing stop
                    lowest_since_entry = close_val
        elif position == 1:
            # Update highest high since entry for trailing stop
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit long: price drops 2.5*ATR from highest high OR trend changes to downtrend
            trailing_stop = highest_since_entry - (2.5 * atr_val)
            exit_condition = (close_val < trailing_stop) or (close_val < ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest low since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit short: price rises 2.5*ATR from lowest low OR trend changes to uptrend
            trailing_stop = lowest_since_entry + (2.5 * atr_val)
            exit_condition = (close_val > trailing_stop) or (close_val > ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeATR_Stop"
timeframe = "4h"
leverage = 1.0