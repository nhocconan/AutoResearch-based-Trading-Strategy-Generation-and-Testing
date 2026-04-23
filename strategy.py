#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R3 AND close > 1d EMA34 AND volume > 1.5x 20-period average.
Short when price breaks below S3 AND close < 1d EMA34 AND volume > 1.5x 20-period average.
Exit when price reverts to H3/L3 level or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
6h timeframe balances trade frequency with signal reliability, while 1d EMA34 provides robust higher-timeframe trend
filter that works in both bull and bear markets by avoiding counter-trend breakouts.
Camarilla pivot levels from 1d provide institutional support/resistance levels with proven effectiveness.
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
    
    # Load 6h data for Camarilla calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate previous 6h bar's OHLC for Camarilla levels
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    prev_close[0] = close_6h[0]
    
    # Calculate Camarilla levels for 6h timeframe
    range_6h = prev_high - prev_low
    camarilla_h3 = prev_close + range_6h * 1.1 / 4
    camarilla_l3 = prev_close - range_6h * 1.1 / 4
    camarilla_h4 = prev_close + range_6h * 1.1 / 2
    camarilla_l4 = prev_close - range_6h * 1.1 / 2
    
    # Align 6h Camarilla levels to 6h timeframe
    h3_6h_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h3)
    l3_6h_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l3)
    h4_6h_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h4)
    l4_6h_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l4)
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 6h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_6h_aligned[i]) or np.isnan(l3_6h_aligned[i]) or 
            np.isnan(h4_6h_aligned[i]) or np.isnan(l4_6h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 AND close > 1d EMA34 AND volume spike
            if (price > h3_6h_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S3 AND close < 1d EMA34 AND volume spike
            elif (price < l3_6h_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to H3 level or ATR stoploss
                if price <= h3_6h_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_6h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to L3 level or ATR stoploss
                if price >= l3_6h_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_6h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0