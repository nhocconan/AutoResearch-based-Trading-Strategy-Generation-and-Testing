#!/usr/bin/env python3
"""
1d_1w_Donchian20_VolumeBreakout_ATRStop_v2
Hypothesis: Daily Donchian(20) breakout with weekly trend filter (EMA50) and volume confirmation.
ATR-based stoploss and take-profit levels manage risk. Designed for low trade frequency
(7-25/year) to minimize fee drag while capturing major trends in BTC/ETH.
Works in bull markets via breakouts and in bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-day)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (no shift needed as get_htf_data returns completed bars)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # ATR(14) for volatility filtering and stoploss
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    volume_ma_20 = pd.Series(prices['volume']).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        atr = atr_14_aligned[i]
        vol_ma = volume_ma_20[i]
        
        # Volume filter
        volume_ok = volume > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above upper Donchian with weekly uptrend and volume
            if (price > upper_20_aligned[i] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and  # Weekly EMA rising
                volume_ok):
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_at_entry = atr
            # Short entry: price breaks below lower Donchian with weekly downtrend and volume
            elif (price < lower_20_aligned[i] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and  # Weekly EMA falling
                  volume_ok):
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_at_entry = atr
        
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            # Stoploss: 2 * ATR below entry
            if price <= entry_price - 2.0 * atr_at_entry:
                exit_signal = True
            # Take profit: 4 * ATR above entry
            elif price >= entry_price + 4.0 * atr_at_entry:
                exit_signal = True
            # Trend reversal: weekly EMA turns down
            elif ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:
                exit_signal = True
            # Donchian breakdown: price breaks below lower band
            elif price < lower_20_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            # Stoploss: 2 * ATR above entry
            if price >= entry_price + 2.0 * atr_at_entry:
                exit_signal = True
            # Take profit: 4 * ATR below entry
            elif price <= entry_price - 4.0 * atr_at_entry:
                exit_signal = True
            # Trend reversal: weekly EMA turns up
            elif ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]:
                exit_signal = True
            # Donchian breakout: price breaks above upper band
            elif price > upper_20_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Donchian20_VolumeBreakout_ATRStop_v2"
timeframe = "1d"
leverage = 1.0