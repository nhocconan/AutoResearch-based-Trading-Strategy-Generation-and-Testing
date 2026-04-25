#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_ChopFilter_v1
Hypothesis: Camarilla H3/L3 breakout with 1d EMA34 trend filter and 4h choppiness regime filter.
Only trade when CHOP(14) > 61.8 (ranging market) to fade false breakouts in ranging conditions
and CHOP < 38.2 (trending market) to ride strong breaks. Targets 20-40 trades/year to avoid fee drag.
Uses discrete position sizing (0.0, ±0.25) to minimize churn. Works in both bull and bear markets
by combining mean reversion in ranges with trend following in strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels (H3, L3) from previous day
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H3 = close + (high - low) * 1.1/4, L3 = close - (high - low) * 1.1/4
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(highest(high,14) - lowest(low,14))) / log10(14)
    # Simplified: CHOP = 100 * log10(atr_sum / (max_high - min_low)) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    chop = np.where(range_14 > 0, 100 * np.log10(atr_sum) / np.log10(range_14) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d Camarilla, 1d EMA34, ATR(14) for CHOP
    start_idx = max(1, 34, 14 + 13)  # 1d data + EMA34 + ATR(14) + CHOP window
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 + 1d uptrend + (CHOP < 38.2 for trend OR CHOP > 61.8 for mean reversion fade)
            # In ranging market (CHOP > 61.8): fade breakout -> short at H3, long at L3
            # In trending market (CHOP < 38.2): follow breakout -> long at H3, short at L3
            if chop[i] < 38.2:  # trending market
                long_setup = (close[i] > camarilla_h3_aligned[i]) and (close[i] > ema_34_1d_aligned[i])
                short_setup = (close[i] < camarilla_l3_aligned[i]) and (close[i] < ema_34_1d_aligned[i])
            else:  # ranging market (CHOP > 61.8) or transitional
                long_setup = (close[i] < camarilla_l3_aligned[i]) and (close[i] > ema_34_1d_aligned[i])  # mean reversion long at L3
                short_setup = (close[i] > camarilla_h3_aligned[i]) and (close[i] < ema_34_1d_aligned[i])  # mean reversion short at H3
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below L3 OR 1d trend turns down
            if (close[i] < camarilla_l3_aligned[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above H3 OR 1d trend turns up
            if (close[i] > camarilla_h3_aligned[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0