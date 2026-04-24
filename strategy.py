#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout + 1w EMA50 trend + volume confirmation + ATR stoploss.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when close breaks above Camarilla H3 AND price > 1w EMA50 AND volume > 1.5 * 12h volume MA(20);
         Short when close breaks below Camarilla L3 AND price < 1w EMA50 AND volume > 1.5 * 12h volume MA(20).
- Exit: ATR-based stoploss (2.0 * ATR(14)) and Camarilla pivot reversion for profit-taking.
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla levels from daily OHLC for structure, volume confirmation for participation,
  1w EMA50 trend filter to avoid counter-trend trades, and ATR for risk management.
- Works in both bull (breakouts with trend) and bear (mean-reversion from extremes) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume MA(20) and ATR(14)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume MA(20)
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 12h
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_12h[0] - low_12h[0]], tr])  # first TR is high-low
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Camarilla levels (H3, L3, pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # Pivot = (high + low + close)/3
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4.0
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4.0
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(vol_ma_12h[i]) or 
            np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr_12h[i]
        
        # Volume confirmation: 1.5x threshold for strict entry
        vol_confirm = curr_volume > 1.5 * vol_ma_12h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above Camarilla H3 AND price > 1w EMA50 (uptrend)
                if curr_close > camarilla_h3_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Close breaks below Camarilla L3 AND price < 1w EMA50 (downtrend)
                elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.0 * ATR below entry
            stoploss = entry_price - 2.0 * curr_atr
            # Profit take: close below Camarilla pivot (mean reversion)
            if curr_close < stoploss or curr_close < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.0 * ATR above entry
            stoploss = entry_price + 2.0 * curr_atr
            # Profit take: close above Camarilla pivot (mean reversion)
            if curr_close > stoploss or curr_close > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1wEMA50_Trend_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0