#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop
Hypothesis: Camarilla pivot levels (R1, S1) from 1d HTF act as intraday support/resistance on 4h.
Breakouts above R1 or below S1 with volume confirmation (>1.3x 20-period average) and 
choppiness regime filter (CHOP > 50 = ranging market) capture mean-reversion failures 
that often reverse in bear markets. ATR(14) stoploss at 2.0x ATR manages risk. 
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag 
and work in ranging/volatile markets like 2025 BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Load HTF data ONCE before loop (1d for Camarilla pivot) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (R1, S1) from previous 1d candle
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    rng = high_1d - low_1d
    r1 = close_1d + 1.1 * rng / 12
    s1 = close_1d - 1.1 * rng / 12
    
    # Align to 4h timeframe (wait for 1d candle to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h Indicators (primary timeframe) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * vol_ma
    
    # Choppiness Index regime filter (CHOP > 50 = ranging market)
    chop_period = 14
    atr1 = pd.Series(high - low)
    atr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    atr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([atr1, atr2, atr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(chop_period)
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(volume_threshold[i]) or np.isnan(chop[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + chop > 50 (ranging)
            if price > r1_aligned[i] and volume[i] > volume_threshold[i] and chop[i] > 50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + chop > 50 (ranging)
            elif price < s1_aligned[i] and volume[i] > volume_threshold[i] and chop[i] > 50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below R1 (breakout failed)
            elif price < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above S1 (breakout failed)
            elif price > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop"
timeframe = "4h"
leverage = 1.0