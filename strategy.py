#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4-hour ATR volatility filter and 1-hour momentum breakout.
# Uses 4h ATR(14) normalized by price to filter for high volatility regimes (ATR/price > 0.02).
# In high volatility, enters long on close > highest high of last 20 periods, short on close < lowest low.
# Uses volume confirmation (volume > 1.5x 20-period average) to avoid false breakouts.
# Designed to work in both bull and bear markets by capturing volatility expansion breakouts.
# Position size fixed at 0.20 (20%) to manage drawdown. Target: 20-50 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for volatility filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h ATR(14) for volatility filter
    atr_len = 14
    if len(df_4h) < atr_len:
        return np.zeros(n)
    
    # Calculate True Range
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_4h = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 4h close price for normalization
    price_4h = close_4h
    price_4h_aligned = align_htf_to_ltf(prices, df_4h, price_4h)
    
    # Volatility ratio: ATR/price > 0.02 indicates high volatility regime
    vol_ratio = atr_4h_aligned / price_4h_aligned
    high_vol = vol_ratio > 0.02
    
    # 1h Donchian channel breakout (20 periods)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(atr_len*2, lookback, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vol_ratio[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter
        if not high_vol[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if volume_confirmed:
                # Breakout above highest high
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = position_size
                # Breakdown below lowest low
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel or volatility drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            vol_drop = vol_ratio[i] < 0.015  # Exit when volatility contracts
            if (close[i] < midpoint or vol_drop):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to midpoint or volatility drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            vol_drop = vol_ratio[i] < 0.015
            if (close[i] > midpoint or vol_drop):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_ATR_Vol_Breakout_Donchian_Volume"
timeframe = "1h"
leverage = 1.0