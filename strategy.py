#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_V1
Hypothesis: 12h Camarilla R1/S1 breakout with 1d HTF trend filter (price > EMA34 for long bias, < EMA34 for short bias)
captures strong directional moves. Volume confirmation (>1.5x 20-period average) filters weak breakouts. 
Choppiness regime filter (CHOP > 61.8 = ranging, < 38.2 = trending) avoids false signals in sideways markets.
Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag and work in both bull/bear markets 
via HTF trend alignment, volume confirmation, and regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Camarilla Pivot Levels (based on previous 12h bar)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # Using previous bar's OHLC to avoid look-ahead
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_range = prev_high - prev_low
    r1 = prev_close + 1.1 * prev_range / 12
    s1 = prev_close - 1.1 * prev_range / 12
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop = np.where(chop_denominator > 0, 100 * np.log10(atr_14 / chop_denominator) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_threshold[i]) 
            or np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + trending market + long HTF bias
            if price > r1[i] and volume_12h[i] > volume_threshold[i] and chop[i] < 38.2 and price > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + trending market + short HTF bias
            elif price < s1[i] and volume_12h[i] > volume_threshold[i] and chop[i] < 38.2 and price < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below R1 (breakout failed) or market becomes ranging
            elif price < r1[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above S1 (breakout failed) or market becomes ranging
            elif price > s1[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_V1"
timeframe = "12h"
leverage = 1.0