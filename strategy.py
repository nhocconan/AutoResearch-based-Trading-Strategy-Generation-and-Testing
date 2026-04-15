#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
# Long when price breaks above 20-period 4h Donchian high, 1d ATR > 0.5% of price, and volume > 1.5x 20-period average.
# Short when price breaks below 20-period 4h Donchian low under same conditions.
# Uses discrete position sizing (0.25) to limit fee drag. Designed to work in both bull (breakouts) and bear (breakdowns) markets.
# Target: 20-50 trades/year to avoid overtrading.

name = "4h_Donchian20_ATR_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = max(lookback, 20)
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when daily ATR is elevated (> 0.5% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: price breaks above Donchian high
        if close[i] > highest_high[i] and vol_filter and vol_confirm:
            signals[i] = 0.25
            
        # Short: price breaks below Donchian low
        elif close[i] < lowest_low[i] and vol_filter and vol_confirm:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals