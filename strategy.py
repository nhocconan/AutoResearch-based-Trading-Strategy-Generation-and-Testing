#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_HTFTrend_ChopFilter_V3
Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1/S1) for breakout entries,
filtered by 1w EMA34 trend and 4h choppiness regime (CHOP > 50 = range, < 50 = trend).
Enter long when price breaks above 1d R1 with 1w uptrend and trending market (CHOP < 50).
Enter short when price breaks below 1d S1 with 1w downtrend and trending market.
Exit on ATR(14) trailing stop (2.0*ATR) or opposite level break.
Uses tighter ATR stop (2.0) and volume confirmation to reduce trades vs V2.
Target: 15-30 trades/year (~60-120 total over 4 years) to minimize fee drag.
Works in bull/bear via HTF trend alignment and regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots, 1w for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align to 4h timeframe (use previous completed daily bar)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1w EMA34 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume confirmation: 20-period volume SMA
    vol_sma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) - 14 period
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid div by zero
    
    # ATR (14-period) for stoploss
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(chop[i]) or np.isnan(atr[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        volume = volume_4h[i]
        
        if position == 0:
            # Volume confirmation: current volume > 1.2 * 20-period average
            volume_confirm = volume > 1.2 * vol_sma_20[i]
            
            # Long conditions: price > 1d R1, 1w uptrend, trending market (CHOP < 50), volume confirm
            long_breakout = price > r1_1d_aligned[i]
            long_trend = price > ema_34_1w_aligned[i]
            long_regime = chop[i] < 50
            
            # Short conditions: price < 1d S1, 1w downtrend, trending market
            short_breakout = price < s1_1d_aligned[i]
            short_trend = price < ema_34_1w_aligned[i]
            short_regime = chop[i] < 50
            
            # Entry logic
            if long_breakout and long_trend and long_regime and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and short_regime and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 1d S1 (support broken)
            elif price < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 1d R1 (resistance broken)
            elif price > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_HTFTrend_ChopFilter_V3"
timeframe = "4h"
leverage = 1.0