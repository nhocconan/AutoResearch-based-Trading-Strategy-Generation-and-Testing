#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel (20) breakout with volume confirmation and 1h EMA(50) trend filter.
Long when price breaks above upper band with volume > 2x 1h average volume and close > 1h EMA(50);
Short when price breaks below lower band with volume > 2x 1h average volume and close < 1h EMA(50).
Exit on opposite band touch or 1.5x ATR stop. Designed for 15-25 trades/year to minimize fee drag.
Works in bull markets via upward breakouts and in bear via downward breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1h data ONCE before loop for EMA and volume average
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) on 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    upper_band = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 1h EMA(50) for trend filter
    close_1h = df_1h['close'].values
    ema_50 = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1h, ema_50)
    
    # 1h volume average (20-period)
    vol_1h = df_1h['volume'].values
    vol_avg_20 = pd.Series(vol_1h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1h, vol_avg_20)
    
    # 1h volume (current) aligned to 4h
    vol_1h_current = df_1h['volume'].values
    vol_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_1h_current)
    
    # ATR for stop (14-period on 4h)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(vol_1h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        
        # Current 1h volume aligned to 4h
        vol_1h_current = vol_1h_aligned[i]
        
        if position == 0:
            # Enter long: break above upper band with volume surge and close > 1h EMA50
            if (price_high > upper_band[i] and 
                vol_1h_current > 2.0 * vol_avg_20_aligned[i] and
                price_close > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower band with volume surge and close < 1h EMA50
            elif (price_low < lower_band[i] and 
                  vol_1h_current > 2.0 * vol_avg_20_aligned[i] and
                  price_close < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite band touch or 1.5x ATR stop
            exit_signal = False
            
            if position == 1:
                # Exit long: touch lower band OR price < entry - 1.5*ATR
                if price_low < lower_band[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use upper band as entry level for long
                    entry_level = upper_band[i-1] if i >= 1 else upper_band[0]
                    if price_close < entry_level - 1.5 * atr[i]:
                        exit_signal = True
            elif position == -1:
                # Exit short: touch upper band OR price > entry + 1.5*ATR
                if price_high > upper_band[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use lower band as entry level for short
                    entry_level = lower_band[i-1] if i >= 1 else lower_band[0]
                    if price_close > entry_level + 1.5 * atr[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume2x_1hEMA50_Trend"
timeframe = "4h"
leverage = 1.0