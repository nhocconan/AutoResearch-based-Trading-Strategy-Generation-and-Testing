#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR-based stoploss.
Long when price breaks above 20-period Donchian upper channel AND close > 1d EMA50 (uptrend) AND volume > 1.5x average.
Short when price breaks below 20-period Donchian lower channel AND close < 1d EMA50 (downtrend) AND volume > 1.5x average.
Uses discrete position sizing (0.25) to minimize fee churn. ATR stoploss exits when price moves against position by 2.5x ATR.
Designed to capture strong trending moves while filtering false breakouts in ranging markets.
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
    
    # Load 4h data for Donchian channel calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h - using previous bar's data to avoid look-ahead
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    # Use previous bar's high/low for channel calculation (avoid look-ahead)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    
    upper_20 = rolling_max(prev_high_4h, 20)
    lower_20 = rolling_min(prev_low_4h, 20)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for stoploss - ONCE before loop
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        atr_val = atr_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 1d EMA50 (uptrend) AND volume confirmation
            if (price > upper_val and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                stop_price = price - 2.5 * atr_val
            # Short: price breaks below Donchian lower AND price < 1d EMA50 (downtrend) AND volume confirmation
            elif (price < lower_val and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                stop_price = price + 2.5 * atr_val
        else:
            # Check stoploss
            if position == 1 and price < stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                stop_price = 0.0
            elif position == -1 and price > stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                stop_price = 0.0
            else:
                # Trail stoploss for favorable moves
                if position == 1:
                    # Trail stop up as price makes new highs
                    new_stop = price - 2.5 * atr_val
                    if new_stop > stop_price:
                        stop_price = new_stop
                else:  # position == -1
                    # Trail stop down as price makes new lows
                    new_stop = price + 2.5 * atr_val
                    if new_stop < stop_price:
                        stop_price = new_stop
                
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA50_Volume_ATR_Stop"
timeframe = "4h"
leverage = 1.0