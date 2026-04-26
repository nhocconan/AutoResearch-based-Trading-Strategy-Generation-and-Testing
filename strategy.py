#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>2.0x 20-bar MA). Uses ATR-based stoploss (2.5x ATR) for risk control. Designed for 4h timeframe to achieve ~20-50 trades/year. Works in bull/bear markets by following 1d trend while using Donchian structure for breakouts. Volume spike filter reduces false breakouts. ATR stoploss manages risk during volatile periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR for volatility and stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for Donchian/vol, 50 for 1d EMA, 14 for ATR)
    start_idx = max(lookback, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 1:
                signals[i] = base_size
            elif position == -1:
                signals[i] = -base_size
            else:
                signals[i] = 0.0
            continue
        
        close_val = close[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        if position == 0:
            # Flat - look for Donchian breakout in trend direction with volume spike
            long_entry = (close_val > highest_high_val) and bullish_1d and vol_spike
            short_entry = (close_val < lowest_low_val) and bearish_1d and vol_spike
            
            if long_entry:
                signals[i] = base_size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -base_size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position - check for exit conditions
            signals[i] = base_size
            
            # Exit on Donchian reversal (close below lowest low of lookback period)
            if close_val < lowest_low_val:
                signals[i] = 0.0
                position = 0
            # Exit on trend change
            elif not bullish_1d:
                signals[i] = 0.0
                position = 0
            # ATR-based stoploss
            elif close_val < entry_price - (2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short position - check for exit conditions
            signals[i] = -base_size
            
            # Exit on Donchian reversal (close above highest high of lookback period)
            if close_val > highest_high_val:
                signals[i] = 0.0
                position = 0
            # Exit on trend change
            elif not bearish_1d:
                signals[i] = 0.0
                position = 0
            # ATR-based stoploss
            elif close_val > entry_price + (2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0