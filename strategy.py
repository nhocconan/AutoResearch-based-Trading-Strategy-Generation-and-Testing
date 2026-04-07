#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly volume filter and ATR volatility scaling
# Uses Donchian(20) channels for breakout signals, weekly average volume to confirm
# institutional interest, and ATR-based volatility scaling to reduce position size
# during high volatility periods. Designed for low trade frequency (target: 15-25 trades/year)
# to minimize fee drag while capturing major trends in both bull and bear markets.

name = "daily_donchian20_weekly_vol_scaled_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly average volume
    vol_1w = df_1w['volume'].values
    vol_avg_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    # ATR(20) for volatility scaling
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: scale position size inversely with volatility
        vol_ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        vol_scale = np.clip(1.0 / vol_ratio, 0.5, 1.0)  # scale between 0.5 and 1.0
        base_size = 0.25
        
        # Volume confirmation: current volume > weekly average volume
        vol_confirm = volume[i] > vol_avg_1w_aligned[i]
        
        # Donchian breakout signals
        long_breakout = close[i] > highest_high[i-1]  # break above previous high
        short_breakout = close[i] < lowest_low[i-1]   # break below previous low
        
        # Long conditions: bullish breakout with volume confirmation
        if long_breakout and vol_confirm:
            signals[i] = base_size * vol_scale
        # Short conditions: bearish breakout with volume confirmation
        elif short_breakout and vol_confirm:
            signals[i] = -base_size * vol_scale
        else:
            signals[i] = 0.0
    
    return signals