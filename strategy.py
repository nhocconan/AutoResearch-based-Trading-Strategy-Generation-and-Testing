#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above R3 AND close > 12h EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below S3 AND close < 12h EMA50 AND volume > 1.5x 20-period average.
Exit when price reverts to Pivot Point (PP) or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Camarilla levels provide intraday support/resistance that work in ranging markets, while 12h EMA50 filters for higher-timeframe trend alignment.
Volume spike confirms institutional participation. This combination should work in both bull and bear markets by avoiding counter-trend breakouts.
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
    
    # Calculate Camarilla levels for 6h timeframe (using previous bar's data)
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    pp_6h = (high_6h + low_6h + close_6h) / 3.0
    range_6h = high_6h - low_6h
    r3_6h = close_6h + range_6h * 1.1 / 2.0
    s3_6h = close_6h - range_6h * 1.1 / 2.0
    
    # Align 6h Camarilla levels to 6h timeframe (no additional delay needed)
    pp_6h_aligned = align_htf_to_ltf(prices, df_6h, pp_6h)
    r3_6h_aligned = align_htf_to_ltf(prices, df_6h, r3_6h)
    s3_6h_aligned = align_htf_to_ltf(prices, df_6h, s3_6h)
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
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
        if (np.isnan(pp_6h_aligned[i]) or np.isnan(r3_6h_aligned[i]) or 
            np.isnan(s3_6h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 AND close > 12h EMA50 AND volume spike
            if (price > r3_6h_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S3 AND close < 12h EMA50 AND volume spike
            elif (price < s3_6h_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Pivot Point or ATR stoploss
                if price <= pp_6h_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_6h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Pivot Point or ATR stoploss
                if price >= pp_6h_aligned[i]:
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

name = "6H_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0