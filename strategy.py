#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 Breakout + 12h EMA34 Trend + Volume Spike + Choppiness Filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when close breaks above H3 level AND price > 12h EMA34 AND volume > 2.0 * 4h volume MA(20) AND chop < 61.8;
         Short when close breaks below L3 level AND price < 12h EMA34 AND volume > 2.0 * 4h volume MA(20) AND chop < 61.8.
- Exit: Long exits when close crosses below L3 level; Short exits when close crosses above H3 level.
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot levels from prior 4h for precise S/R, volume confirmation for participation,
  EMA34 trend filter to avoid counter-trend trades, and choppiness filter to avoid ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot (prior 4h OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels from prior 4h OHLC
    # Camarilla: H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    camarilla_H3 = close_4h + 1.125 * (high_4h - low_4h)
    camarilla_L3 = close_4h - 1.125 * (high_4h - low_4h)
    
    # Align 4h indicators to 4h timeframe (no shift needed as we use prior completed 4h bar)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L3)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 4h (using ATR and high-low range)
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[tr1[0]], tr2])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = max_high14 - min_low14
    
    # Avoid division by zero
    chop = np.where(range14 > 0, 100 * np.log10(atr14 * 14 / range14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 needs 34, volume MA needs 20, CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_chop = chop[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        # Choppiness filter: only trade in trending markets (CHOP < 61.8)
        chop_filter = curr_chop < 61.8
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and chop_filter:
                # Long: Close breaks above H3 AND price > 12h EMA34 (uptrend)
                if curr_close > camarilla_H3_aligned[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below L3 AND price < 12h EMA34 (downtrend)
                elif curr_close < camarilla_L3_aligned[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below L3
            if curr_close < camarilla_L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close crosses above H3
            if curr_close > camarilla_H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0