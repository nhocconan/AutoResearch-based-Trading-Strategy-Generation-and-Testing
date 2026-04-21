#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with volume confirmation and trend filter.
Long when price breaks above 20-period high with volume > 2x average and close > daily EMA(50);
Short when price breaks below 20-period low with volume > 2x average and close < daily EMA(50).
Exit on opposite Donchian band touch or 1.5x ATR stop. Designed for 15-25 trades/year to minimize fee drag.
Works in bull markets via upward breakouts and in bear via downward breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h
    high_max = prices['high'].rolling(window=20, min_periods=20).max().values
    low_min = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Daily EMA(50) for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # ATR for stop (14-period on 4h)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        
        # Current daily volume aligned to 4h
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_1d_current = vol_1d_aligned[i]
        
        if position == 0:
            # Enter long: break above Donchian high with volume surge and close > daily EMA50
            if (price_high > high_max[i] and 
                vol_1d_current > 2.0 * vol_ma_20_aligned[i] and
                price_close > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low with volume surge and close < daily EMA50
            elif (price_low < low_min[i] and 
                  vol_1d_current > 2.0 * vol_ma_20_aligned[i] and
                  price_close < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite Donchian band touch or 1.5x ATR stop
            exit_signal = False
            
            if position == 1:
                # Exit long: touch Donchian low OR price < entry - 1.5*ATR
                if price_low < low_min[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use Donchian high as entry level for long
                    entry_level = high_max[i-1] if i >= 1 else high_max[0]
                    if price_close < entry_level - 1.5 * atr[i]:
                        exit_signal = True
            elif position == -1:
                # Exit short: touch Donchian high OR price > entry + 1.5*ATR
                if price_high > high_max[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use Donchian low as entry level for short
                    entry_level = low_min[i-1] if i >= 1 else low_min[0]
                    if price_close > entry_level + 1.5 * atr[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume2x_EMA50Trend_ATR1.5x"
timeframe = "4h"
leverage = 1.0