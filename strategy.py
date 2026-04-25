#!/usr/bin/env python3
"""
12h_WilliamsAlligator_1dTrend_VolumeBreakout
Hypothesis: 12h Williams Alligator (jaw/teeth/lips) with 1d EMA50 trend filter and volume confirmation (>1.5x 24-bar avg). 
Enters long when Alligator is bullish (lips > teeth > jaw) AND price breaks above Alligator lips in 1d uptrend.
Enters short when Alligator is bearish (lips < teeth < jaw) AND price breaks below Alligator lips in 1d downtrend.
Uses ATR-based stoploss (2.0x ATR) and discrete sizing (0.25) to limit fee churn. Designed for 12h timeframe with ~12-37 trades/year.
Works in bull/bear by following 1d trend filter and requiring Alligator alignment + volume spike for entries.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (using SMAs)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA calculation using EMA as approximation (common implementation)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # ATR for stoploss (using 14 periods)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current volume > 1.5x 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need sufficient data for Alligator (13), ATR (14), volume MA (24), 1d EMA (50)
    start_idx = max(24, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Alligator conditions
        alligator_bullish = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_bearish = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Price relative to Alligator lips
        price_above_lips = curr_close > lips[i]
        price_below_lips = curr_close < lips[i]
        
        if position == 0:
            # Long: Alligator bullish AND price breaks above lips AND 1d uptrend AND volume spike
            bullish_entry = alligator_bullish and price_above_lips and \
                           (close_1d[i] > ema_50_1d_aligned[i]) and \
                           volume_spike[i]
            # Short: Alligator bearish AND price breaks below lips AND 1d downtrend AND volume spike
            bearish_entry = alligator_bearish and price_below_lips and \
                           (close_1d[i] < ema_50_1d_aligned[i]) and \
                           volume_spike[i]
            
            if bullish_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - (2.0 * atr[i])
            elif bearish_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Alligator turns bearish OR price breaks below lips OR stoploss hit OR 1d trend turns down
            if (not alligator_bullish) or \
               (curr_close < lips[i]) or \
               (curr_close < atr_stop) or \
               (close_1d[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Alligator turns bullish OR price breaks above lips OR stoploss hit OR 1d trend turns up
            if (not alligator_bearish) or \
               (curr_close > lips[i]) or \
               (curr_close > atr_stop) or \
               (close_1d[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WilliamsAlligator_1dTrend_VolumeBreakout"
timeframe = "12h"
leverage = 1.0