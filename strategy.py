#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 breakouts on 12h timeframe with 1d EMA34 trend filter and volume confirmation
captures strong momentum moves while reducing false breakouts. The chop filter avoids ranging markets.
Designed to work in both bull and bear markets by following higher timeframe trend. Targets 12-30 trades/year
to stay under the 200 trade hard limit for 12h and minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate daily Camarilla levels (using prior day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Chop filter: avoid ranging markets (Choppiness Index > 61.8)
    # Use 14-period chop on close prices
    close_series = pd.Series(close)
    atr_14 = close_series.rolling(window=14, min_periods=14).apply(
        lambda x: np.max(np.abs(np.diff(x, prepend=x[0]))), raw=True
    ).rolling(window=14, min_periods=14).sum()
    true_range = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                                     np.abs(low - np.roll(close, 1))), index=prices.index)
    atr_14 = true_range.rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(atr_14 / (np.max(high, axis=0) - np.min(low, axis=0))) / np.log10(14)
    chop = np.where((np.max(high, axis=0) - np.min(low, axis=0)) != 0, chop, 50)
    chopping_market = chop > 61.8  # True when choppy/ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(chopping_market[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_chopping = chopping_market[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals (only in non-choppy markets)
            # Long: price breaks above Camarilla H3 AND bullish bias AND volume spike AND not chopping
            long_entry = (curr_high > camarilla_h3_aligned[i]) and bullish_bias and vol_spike and (not is_chopping)
            # Short: price breaks below Camarilla L3 AND bearish bias AND volume spike AND not chopping
            short_entry = (curr_low < camarilla_l3_aligned[i]) and bearish_bias and vol_spike and (not is_chopping)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla L3 (mean reversion) OR loss of bullish bias OR market becomes choppy
            if (curr_low < camarilla_l3_aligned[i]) or (curr_close < ema_1d_aligned[i]) or is_chopping:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla H3 (mean reversion) OR loss of bearish bias OR market becomes choppy
            if (curr_high > camarilla_h3_aligned[i]) or (curr_close > ema_1d_aligned[i]) or is_chopping:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3_L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0