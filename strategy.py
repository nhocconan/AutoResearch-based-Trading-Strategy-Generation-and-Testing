#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA200 trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA200 AND volume > 1.5x average.
Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND price < 1d EMA200 AND volume > 1.5x average.
Exit when Elder Ray momentum reverses (Bull Power <= 0 for long, Bear Power >= 0 for short) or ATR-based stoploss.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Elder Ray measures bull/bear power via EMA13, effective in both trending and ranging markets when combined with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Elder Ray calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate EMA13 for Elder Ray on 6h data
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    # Calculate ATR(14) on 6h data for stoploss
    tr1 = np.maximum(high_6h - low_6h, np.abs(high_6h - np.roll(close_6h, 1)))
    tr2 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_6h[0] - low_6h[0]  # first bar
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 6h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 6h close for price comparison
        price_6h = close_6h[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA200 AND volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                price_6h > ema200_1d_aligned[i] and 
                volume_6h[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price_6h
            # Short: Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND price < 1d EMA200 AND volume confirmation
            elif (bear_power[i] < 0 and bull_power[i] < 0 and 
                  price_6h < ema200_1d_aligned[i] and 
                  volume_6h[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price_6h
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power <= 0 (momentum lost) OR ATR stoploss
                if bull_power[i] <= 0:
                    exit_signal = True
                elif price_6h < entry_price - 2.5 * atr_6h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bear Power >= 0 (momentum lost) OR ATR stoploss
                if bear_power[i] >= 0:
                    exit_signal = True
                elif price_6h > entry_price + 2.5 * atr_6h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dEMA200_Volume_ATRStop"
timeframe = "6h"
leverage = 1.0