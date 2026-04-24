#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout + 1d EMA50 trend + volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when close breaks above Camarilla H3 AND price > 1d EMA50 AND volume > 1.5 * 12h volume MA(20);
         Short when close breaks below Camarilla L3 AND price < 1d EMA50 AND volume > 1.5 * 12h volume MA(20).
- Exit: Close below/above Camarilla pivot point (PP) for profit-taking, ATR-based stoploss (2.5 * ATR(20)).
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla levels from 1d OHLC for structure, volume confirmation for participation,
  EMA50 trend filter to avoid counter-trend trades, and ATR for risk management.
- Designed to work in both bull and bear markets by only taking trades in direction of 1d trend.
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
    
    # Get 12h data for Camarilla levels (using prior completed 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels from prior 12h OHLC
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    # H3 = C + (H-L) * 1.1/4, L3 = C - (H-L) * 1.1/4
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    h3_12h = close_12h + range_12h * 1.1 / 4.0
    l3_12h = close_12h - range_12h * 1.1 / 4.0
    
    # Align 12h Camarilla levels to 12h timeframe (no shift needed as we use prior completed 12h bar)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 12h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # EMA50 needs 50, volume MA needs 20, ATR needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr20[i]
        
        # Volume confirmation: 1.5x threshold for strict entry
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above Camarilla H3 AND price > 1d EMA50 (uptrend)
                if curr_close > h3_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Close breaks below Camarilla L3 AND price < 1d EMA50 (downtrend)
                elif curr_close < l3_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.5 * ATR below entry
            stoploss = entry_price - 2.5 * curr_atr
            # Profit take: close below Camarilla pivot point
            if curr_close < stoploss or curr_close < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.5 * ATR above entry
            stoploss = entry_price + 2.5 * curr_atr
            # Profit take: close above Camarilla pivot point
            if curr_close > stoploss or curr_close > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0