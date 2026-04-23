#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend + volume confirmation + ATR stoploss.
Long when price breaks above Donchian upper band AND close > 1d EMA34 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA34 AND volume > 1.8x 20-period average.
Exit on opposite Donchian band touch or ATR stoploss (2.0x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Adds 1d trend filter to avoid counter-trend trades and improve BTC/ETH performance in both bull/bear markets.
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
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)))
    tr2 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_4h[0] - low_4h[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on 4h data
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND close > 1d EMA34 AND volume spike
            if (price > donchian_upper[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND close < 1d EMA34 AND volume spike
            elif (price < donchian_lower[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches Donchian lower or ATR stoploss
                if price < donchian_lower[i]:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches Donchian upper or ATR stoploss
                if price > donchian_upper[i]:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0