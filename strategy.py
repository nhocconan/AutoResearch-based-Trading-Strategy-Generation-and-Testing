#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrendFilter_RegimeAvoidChop
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d EMA50 trend filter and chop regime avoidance.
Avoids choppy markets (CHOP > 61.8) where breakouts fail. Uses discrete sizing (0.25) to minimize fees.
Works in bull/bear by following 1d trend; avoids whipsaws in ranging markets via chop filter.
Target: 12-30 trades/year. Uses 1d HTF for trend and regime, 12h for entries/exits.
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
    
    # Get 1d data for HTF trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR1) / (n * (HHV - LLV))) / log10(n)
    # Where ATR1 = true range, HHV = highest high over period, LLV = lowest low over period
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hhvl = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llvl = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr1 / (14 * (hhvl - llvl))) / np.log10(14)
    chop_1d = np.where((hhvl - llvl) != 0, chop_raw, 50.0)  # avoid division by zero
    
    # Get 12h data for Camarilla levels (based on previous bar's OHLC)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Camarilla levels on 12h data
    camarilla_r1_12h = close_12h + ((high_12h - low_12h) * 1.1 / 12)
    camarilla_s1_12h = close_12h - ((high_12h - low_12h) * 1.1 / 12)
    camarilla_h4_12h = close_12h + ((high_12h - low_12h) * 1.1 / 2)
    camarilla_l4_12h = close_12h - ((high_12h - low_12h) * 1.1 / 2)
    
    # Align HTF indicators to 12h timeframe (completed 1d bar lag)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d, additional_delay_bars=1)
    
    # Align 12h indicators to 12h timeframe (completed 12h bar lag)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4_12h, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4_12h, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and chop
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: avoid choppy markets (CHOP > 61.8 = ranging)
        in_chop = chop_aligned[i] > 61.8
        
        if position == 0:
            # Look for breakout signals in direction of 1d trend, only if not choppy
            # Long: price breaks above R1 in uptrend (close > EMA50) AND not choppy
            # Short: price breaks below S1 in downtrend (close < EMA50) AND not choppy
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i] and (not in_chop)
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i] and (not in_chop)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla H4 (take profit at resistance)
            exit_signal = close[i] < camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla L4 (take profit at support)
            exit_signal = close[i] > camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrendFilter_RegimeAvoidChop"
timeframe = "12h"
leverage = 1.0