#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian(20) breakout + 4h EMA50 trend filter + volume confirmation.
Long when price breaks above 20-period Donchian upper band AND close > 4h EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below 20-period Donchian lower band AND close < 4h EMA50 AND volume > 1.5x 20-period average.
Exit when price crosses the Donchian middle band (mean reversion) or ATR-based stoploss (2.0x ATR).
Uses discrete position sizing (0.20) to minimize fee churn. Targets 15-37 trades/year per symbol.
Donchian channels provide objective breakout levels, while 4h EMA50 ensures alignment with intermediate trend to avoid counter-trend trades.
Volume confirmation filters weak breakouts. Works in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h data
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Donchian(20) on 1h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 1h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND close > 4h EMA50 AND volume spike
            if (price > donchian_upper[i] and 
                close[i] > ema50_4h_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band AND close < 4h EMA50 AND volume spike
            elif (price < donchian_lower[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses Donchian middle band or ATR stoploss
                if price < donchian_middle[i]:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses Donchian middle band or ATR stoploss
                if price > donchian_middle[i]:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian20_4hEMA50_VolumeConfirm"
timeframe = "1h"
leverage = 1.0