#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1 AND 12h EMA50 rising AND volume > 2x average.
Short when price breaks below S1 AND 12h EMA50 falling AND volume > 2x average.
Exit when price reverts to Camarilla pivot point (PP) or ATR-based stoploss.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
Camarilla pivot levels provide high-probability intraday reversal points; breakouts with volume and trend filter capture strong moves.
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
    
    # Load 4h data for Camarilla pivot calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate previous 4h bar's Camarilla levels
    # PP = (H + L + C) / 3
    # R1 = PP + (H - L) * 1.1 / 12
    # S1 = PP - (H - L) * 1.1 / 12
    # We need the previous completed 4h bar's data
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    PP = (prev_high + prev_low + prev_close) / 3.0
    R1 = PP + (prev_high - prev_low) * 1.1 / 12.0
    S1 = PP - (prev_high - prev_low) * 1.1 / 12.0
    
    # Align 4h Camarilla levels to 15m timeframe (since we're using 4h data on 4h timeframe, no alignment needed)
    # But we need to align to the primary timeframe (4h) - actually, we are on 4h timeframe, so we can use directly
    # However, to be safe and follow MTF rules, we'll load 12h data for trend and align
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate ATR(14) on 12h data for stoploss
    tr1 = np.maximum(high_12h - low_12h, np.abs(high_12h - np.roll(close_12h, 1)))
    tr2 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_12h[0] - low_12h[0]  # first bar
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volume average (20-period) on 12h timeframe
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(PP[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 4h close for price comparison (current bar close)
        price_4h = close_4h[i]
        vol_ma_val = vol_ma_12h_aligned[i]
        
        if position == 0:
            # Check for EMA50 trend direction (rising/falling)
            ema50_rising = ema50_12h_aligned[i] > ema50_12h_aligned[i-1] if i > 0 else False
            ema50_falling = ema50_12h_aligned[i] < ema50_12h_aligned[i-1] if i > 0 else False
            
            # Long: price breaks above R1 AND EMA50 rising AND volume > 2x average
            if (price_4h > R1[i] and ema50_rising and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price_4h
            # Short: price breaks below S1 AND EMA50 falling AND volume > 2x average
            elif (price_4h < S1[i] and ema50_falling and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price_4h
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to PP OR ATR stoploss
                if price_4h <= PP[i]:
                    exit_signal = True
                elif price_4h < entry_price - 2.5 * atr_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to PP OR ATR stoploss
                if price_4h >= PP[i]:
                    exit_signal = True
                elif price_4h > entry_price + 2.5 * atr_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0