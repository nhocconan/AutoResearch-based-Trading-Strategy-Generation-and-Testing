#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND close > 1d EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND close < 1d EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses Camarilla H3/L3 levels (mean reversion zones) or ATR stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
The 1d EMA50 provides a robust trend filter that works across bull/bear regimes, avoiding counter-trend entries.
Volume confirmation at 2.0x ensures only high-momentum breakouts are taken, reducing false signals.
Camarilla levels from 1d timeframe provide institutional support/resistance with high accuracy.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for price action - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for EMA50 trend filter and Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels for 1d timeframe
    # Camarilla: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But for intraday, we use: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    # R3 = close + 1.5*(high-low), S3 = close - 1.5*(high-low)
    # H3/L3 = mean reversion zones, R3/S3 = breakout zones
    hl_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * hl_range  # Mean reversion resistance
    camarilla_l3 = close_1d - 1.1 * hl_range  # Mean reversion support
    camarilla_r3 = close_1d + 1.5 * hl_range  # Breakout resistance
    camarilla_s3 = close_1d - 1.5 * hl_range  # Breakout support
    
    # Align 1d indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate ATR(20) for stoploss on 12h data
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period: no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_favorable_price = 0.0  # For ATR trailing stop logic
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_favorable_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 1d EMA50 AND volume spike
            if (price > camarilla_r3_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                max_favorable_price = price
            # Short: price breaks below Camarilla S3 AND close < 1d EMA50 AND volume spike
            elif (price < camarilla_s3_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                max_favorable_price = price
        else:
            # Update max favorable price for trailing stop
            if position == 1:
                if price > max_favorable_price:
                    max_favorable_price = price
            else:  # position == -1
                if price < max_favorable_price:
                    max_favorable_price = price
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla H3/L3 (mean reversion zones)
            if position == 1 and price < camarilla_h3_aligned[i]:
                exit_signal = True
            elif position == -1 and price > camarilla_l3_aligned[i]:
                exit_signal = True
            
            # ATR-based stoploss: exit if adverse move > 2.5 * ATR from max favorable price
            if not exit_signal:
                if position == 1:
                    adverse_move = max_favorable_price - price
                    if adverse_move > 2.5 * atr_val:
                        exit_signal = True
                else:  # position == -1
                    adverse_move = price - max_favorable_price
                    if adverse_move > 2.5 * atr_val:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_favorable_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_1dEMA50_VolumeConfirm_ATRStop"
timeframe = "12h"
leverage = 1.0