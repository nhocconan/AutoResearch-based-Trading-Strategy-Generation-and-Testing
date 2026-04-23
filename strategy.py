#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volume spike and trend filter.
Long when price breaks above Donchian upper band and ATR(14) > 1.5*ATR(50) with price > 1d EMA50.
Short when price breaks below Donchian lower band and ATR(14) > 1.5*ATR(50) with price < 1d EMA50.
Exit on opposite Donchian break or ATR contraction (<1.2*ATR(50)).
Designed to capture strong momentum bursts with volatility expansion, avoiding false breakouts in low-vol regimes.
ATR ratio filter ensures we only trade during genuine volatility expansion, reducing whipsaws.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ATR(14) and ATR(50) on primary timeframe
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    
    tr2 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr2, np.abs(low[1:] - close[:-1]))
    tr2 = np.concatenate([[np.nan], tr2])
    
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr2).rolling(window=50, min_periods=50).mean().values
    
    # Donchian channels (20-period) on primary timeframe
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        atr14_val = atr14[i]
        atr50_val = atr50[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND ATR expansion AND price > 1d EMA50 (uptrend)
            if (price > donch_high[i] and atr14_val > 1.5 * atr50_val and price > ema50_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND ATR expansion AND price < 1d EMA50 (downtrend)
            elif (price < donch_low[i] and atr14_val > 1.5 * atr50_val and price < ema50_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower OR ATR contraction
                if (price < donch_low[i] or atr14_val < 1.2 * atr50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper OR ATR contraction
                if (price > donch_high[i] or atr14_val < 1.2 * atr50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_ATR_Expansion_1dEMA50"
timeframe = "12h"
leverage = 1.0