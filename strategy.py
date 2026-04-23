#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above R3 AND close > 1w EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below S3 AND close < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when price reverts to Camarilla Pivot point (PP) or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
12h timeframe reduces trade frequency vs 4h while maintaining responsiveness. 1w EMA50 provides
strong long-term trend filter that works in both bull and bear markets. Camarilla R3/S3 levels
provide stronger support/resistance than R1/S1, reducing false breakouts.
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
    
    # Load 12h data for Camarilla calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for 12h timeframe (using previous bar's OHLC)
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # First bar: use current values (will be refined as more data comes)
    prev_high[0] = high_12h[0]
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + camarilla_range * 1.1 / 4.0
    camarilla_s3 = prev_close - camarilla_range * 1.1 / 4.0
    
    # Align 12h Camarilla levels to 12h timeframe (no additional delay needed as they're based on completed bar)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 12h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 AND close > 1w EMA50 AND volume spike
            if (price > camarilla_r3_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S3 AND close < 1w EMA50 AND volume spike
            elif (price < camarilla_s3_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to PP or ATR stoploss
                if price <= camarilla_pp_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_12h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to PP or ATR stoploss
                if price >= camarilla_pp_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_12h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0