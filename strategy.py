#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA20 trend filter and 1d chop regime filter. 
Only trades when price breaks above/below Camarilla levels aligned with 4h trend and 1d market is not too choppy (CHOP < 61.8). 
Volume confirmation (>1.5x 24-bar avg) ensures participation. 
Designed for low trade frequency (15-35/year) to minimize fee drag and work in both bull/bear markets via trend alignment and regime filter.
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
    
    # Get 4h data for HTF trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA20 on 4h close for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels on 4h data (based on previous bar's OHLC)
    camarilla_r1_4h = close_4h + ((high_4h - low_4h) * 1.1 / 12)
    camarilla_s1_4h = close_4h - ((high_4h - low_4h) * 1.1 / 12)
    camarilla_h4_4h = close_4h + ((high_4h - low_4h) * 1.1 / 2)
    camarilla_l4_4h = close_4h - ((high_4h - low_4h) * 1.1 / 2)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chopiness Index (CHOP) on 1d data
    def calculate_chop(high, low, close, window=14):
        atr = []
        for i in range(len(high)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr.append(tr)
        
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if atr_sum[i] > 0 and hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(window)
            else:
                chop[i] = 50.0
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    
    # Align HTF indicators to 1h timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4_4h)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4_4h)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: 1.5x 24-bar average volume (moderate filter)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour.values
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA20 (20) and chop calculation (14) and volume MA (24)
    start_idx = max(20, 14, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema20_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Regime filter: only trade when market is not too choppy (CHOP < 61.8 = trending)
        regime_filter = chop_aligned[i] < 61.8
        
        if position == 0:
            # Look for breakout signals with trend filter and regime filter
            # Long: price breaks above R1 in uptrend (close > EMA20) with volume spike and good regime
            # Short: price breaks below S1 in downtrend (close < EMA20) with volume spike and good regime
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema20_aligned[i]) and volume_spike[i] and regime_filter
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema20_aligned[i]) and volume_spike[i] and regime_filter
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price moves back below Camarilla H4 (take profit at resistance)
            exit_signal = close[i] < camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above Camarilla L4 (take profit at support)
            exit_signal = close[i] > camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime"
timeframe = "1h"
leverage = 1.0