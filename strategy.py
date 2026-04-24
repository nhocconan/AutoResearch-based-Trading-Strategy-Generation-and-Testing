#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX(12) zero-cross with volume spike confirmation and 1d choppiness regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d choppiness index (CHOP) > 61.8 for ranging market (mean reversion), CHOP < 38.2 for trending.
- Entry: Long when TRIX crosses above zero AND volume > 2.0 * 4h volume MA(20) AND CHOP > 61.8 (oversold bounce in range);
         Short when TRIX crosses below zero AND volume > 2.0 * 4h volume MA(20) AND CHOP > 61.8 (overbought fade in range).
- Exit: Opposite TRIX zero-cross or ATR-based stoploss (2.0 * ATR(14)).
- Signal size: 0.25 discrete to control fee drag.
- Uses TRIX momentum for reversals in ranging markets, volume confirmation for participation,
  and 1d choppiness regime to avoid trending markets where mean reversion fails.
- Designed to work in both bull and bear markets via regime filter and tight entry conditions.
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
    
    # Get 4h data for TRIX calculation and volume MA(20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ATR(14) for 4h timeframe
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_4h[0] - low_4h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate TRIX(12,12,12) on 4h close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then / previous value * 100
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix_raw.values
    
    # Get 1d data for choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate choppiness index (CHOP) = 100 * log10(sum(ATR(14)) / log10(n) / (HHV - LLV))
    # where ATR(14) = true range, HHV = highest high over period, LLV = lowest low over period
    tr1d = np.maximum(high_1d[1:] - low_1d[1:], 
                      np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                 np.abs(low_1d[1:] - close_1d[:-1])))
    tr1d = np.concatenate([[high_1d[0] - low_1d[0]], tr1d])
    atr14_1d = pd.Series(tr1d).rolling(window=14, min_periods=14).sum()
    hhv14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    llv14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14_1d / np.log10(14) / (hhv14 - llv14))
    chop = chop.values
    
    # Align 4h TRIX and volume MA, and 1d CHOP to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    atr14_aligned = align_htf_to_ltf(prices, df_4h, atr14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(36, 20, 14, 28)  # TRIX needs ~36, volume MA needs 20, ATR needs 14, CHOP needs 28
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(trix_aligned[i-1]) or  # need previous for crossover
            np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(atr14_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr14_aligned[i]
        curr_chop = chop_aligned[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_4h_aligned[i]
        
        # Choppiness regime: only trade in ranging markets (CHOP > 61.8)
        in_range = curr_chop > 61.8
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and in_range:
                # Long: TRIX crosses above zero (bullish momentum in range)
                if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: TRIX crosses below zero (bearish momentum in range)
                elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.0 * ATR below entry
            stoploss = entry_price - 2.0 * curr_atr
            # Opposite signal: TRIX crosses below zero
            if curr_close < stoploss or (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.0 * ATR above entry
            stoploss = entry_price + 2.0 * curr_atr
            # Opposite signal: TRIX crosses above zero
            if curr_close > stoploss or (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_ZeroCross_VolumeSpike_ChopRange_v1"
timeframe = "4h"
leverage = 1.0