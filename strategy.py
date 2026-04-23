#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume spike.
Long when price breaks above 12h Donchian upper(20) AND close > 1d EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below 12h Donchian lower(20) AND close < 1d EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses 12h Donchian middle line or ATR stoploss (2.5x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Donchian channels provide clear trend-following structure, while 1d EMA50 ensures alignment with daily trend.
Volume confirmation filters weak breakouts. ATR stoploss manages risk. Works in both bull and bear regimes.
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
    
    # Load 12h data for Donchian channels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 12h data
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate ATR(14) on 12h data for stoploss
    tr1 = np.maximum(high_12h - low_12h, np.abs(high_12h - np.roll(close_12h, 1)))
    tr2 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_12h[0] - low_12h[0]  # first bar
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Align 12h indicators to LTF (primary timeframe is 12h, so no alignment needed for 12h indicators)
    # But we need to align to the prices timeframe (which is 12h as per strategy declaration)
    # Since prices is already 12h, we can use the 12h indicators directly
    # However, we need to ensure we're using completed 12h bars, so we shift by 1
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_mid_shifted = np.roll(donchian_mid, 1)
    atr_12h_shifted = np.roll(atr_12h, 1)
    vol_ma_shifted = np.roll(vol_ma, 1)
    
    # First bar has no previous data
    donchian_high_shifted[0] = donchian_high[0]
    donchian_low_shifted[0] = donchian_low[0]
    donchian_mid_shifted[0] = donchian_mid[0]
    atr_12h_shifted[0] = atr_12h[0]
    vol_ma_shifted[0] = vol_ma[0]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_shifted[i]) or np.isnan(donchian_low_shifted[i]) or 
            np.isnan(donchian_mid_shifted[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_shifted[i]) or np.isnan(atr_12h_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_shifted[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper AND close > 1d EMA50 AND volume spike
            if (price > donchian_high_shifted[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 12h Donchian lower AND close < 1d EMA50 AND volume spike
            elif (price < donchian_low_shifted[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below 12h Donchian middle or ATR stoploss
                if price < donchian_mid_shifted[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_12h_shifted[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above 12h Donchian middle or ATR stoploss
                if price > donchian_mid_shifted[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_12h_shifted[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0