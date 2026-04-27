#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeATR_Stop
Hypothesis: Uses 4h Donchian(20) breakouts aligned with 12h EMA50 trend filter and volume confirmation (>1.5x 20-bar avg). 
ATR-based trailing stop (3x ATR) manages risk. Designed for moderate trade frequency (~25-40/year) to avoid fee drag.
Works in bull/bear via trend filter and volatility-based stops.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 12h EMA50 (50), ATR (14), Donchian (20), volume avg (20)
    start_idx = max(50, 14, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_12h_aligned[i]
        atr_val = atr[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 12h trend filter AND volume
            # Long: price breaks above upper channel AND 12h uptrend AND volume
            long_condition = (close_val > upper_channel) and (close_val > ema_val) and vol_conf
            # Short: price breaks below lower channel AND 12h downtrend AND volume
            short_condition = (close_val < lower_channel) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Trail long: exit when price drops to highest_high - 3*ATR
            exit_level = upper_channel - (3.0 * atr_val)
            if close_val <= exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Trail short: exit when price rises to lowest_low + 3*ATR
            exit_level = lower_channel + (3.0 * atr_val)
            if close_val >= exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeATR_Stop"
timeframe = "4h"
leverage = 1.0