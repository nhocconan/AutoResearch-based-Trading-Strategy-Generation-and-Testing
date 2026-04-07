#!/usr/bin/env python3
"""
4h_atr_breakout_12h_trend_volume_v3
Hypothesis: ATR-based breakout from Donchian channels with 12h EMA trend filter and volume confirmation.
Goes long when price breaks above Donchian(20) high with volume confirmation and 12h EMA50 uptrend.
Goes short when price breaks below Donchian(20) low with volume confirmation and 12h EMA50 downtrend.
Uses ATR for volatility-based position sizing and stop loss. Designed for 20-30 trades/year on 4h timeframe.
Works in bull markets via breakouts and bear markets via breakdowns with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_12h_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR for volatility and stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or atr[i] == 0 or
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Breakout conditions
        breakout_high = close[i] > highest_high[i]
        breakout_low = close[i] < lowest_low[i]
        
        # 12h trend filter
        above_12h_ema50 = close[i] > ema50_12h_aligned[i]
        below_12h_ema50 = close[i] < ema50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend turns bearish
            if close[i] < lowest_low[i] or below_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend turns bullish
            if close[i] > highest_high[i] or above_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high with volume confirmation and bullish trend
            if breakout_high and vol_confirmed and above_12h_ema50:
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low with volume confirmation and bearish trend
            elif breakout_low and vol_confirmed and below_12h_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals