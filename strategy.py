#!/usr/bin/env python3
"""
Hypothesis: 4-hour volume-weighted breakout with 12-hour trend filter and volatility filter.
Long when price breaks above Donchian(20) high with volume > 1.5x average volume and 12h EMA50 rising.
Short when price breaks below Donchian(20) low with volume > 1.5x average volume and 12h EMA50 falling.
Exit when price crosses opposite Donchian boundary or EMA50 direction reverses.
Volume confirmation reduces false breakouts; 12h EMA50 filters trend direction; volatility filter avoids choppy markets.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following 12h trend while using 4h breakouts for entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (14-period)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely low volatility (choppy) markets
        vol_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: Price breaks above Donchian high, volume > 1.5x average, 12h EMA50 rising, volatility filter
            if (close[i] > high_20[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, volume > 1.5x average, 12h EMA50 falling, volatility filter
            elif (close[i] < low_20[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and
                  vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian low OR 12h EMA50 starts falling
                if (close[i] < low_20[i] or 
                    ema50_12h_aligned[i] < ema50_12h_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian high OR 12h EMA50 starts rising
                if (close[i] > high_20[i] or 
                    ema50_12h_aligned[i] > ema50_12h_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Volume_Weighted_Breakout_12hEMA50_Trend_VolFilter"
timeframe = "4h"
leverage = 1.0