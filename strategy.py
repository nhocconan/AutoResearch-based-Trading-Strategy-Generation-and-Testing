#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1d EMA50 AND volume > 1.3x average.
Short when price breaks below Donchian lower band AND close < 1d EMA50 AND volume > 1.3x average.
Exit when price crosses Donchian midpoint or ATR-based stoploss (2.5x ATR).
Uses discrete position sizing (0.30) to balance return and drawdown.
Targets 20-40 trades/year per symbol (~80-160 total over 4 years).
Donchian channels provide clear breakout levels, effective in both trending and ranging markets when combined with trend filter and volume confirmation.
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
    
    # Calculate Donchian channels on primary timeframe (4h)
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR(14) on primary timeframe for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 4h close for price comparison
        price = close[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: break above Donchian upper AND price > 1d EMA50 AND volume confirmation
            if (price > highest_high[i] and 
                price > ema50_1d_aligned[i] and 
                volume[i] > 1.3 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: break below Donchian lower AND price < 1d EMA50 AND volume confirmation
            elif (price < lowest_low[i] and 
                  price < ema50_1d_aligned[i] and 
                  volume[i] > 1.3 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses Donchian midpoint OR ATR-based stoploss
                if price < donchian_mid[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses Donchian midpoint OR ATR-based stoploss
                if price > donchian_mid[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_1dEMA50_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0