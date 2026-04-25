#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels act as breakout points on 1d chart. Price breaking above H3 or below L3 with 1d EMA34 trend alignment, volume confirmation, and choppy market filter (CHOP > 61.8) captures momentum moves in both bull and bear markets via trend filter and discrete sizing (0.25). Targets 75-200 trades over 4 years on 4h.
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
    
    # Load 1d data ONCE before loop for Camarilla pivots, EMA34, and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Camarilla pivot levels (based on previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each 1d bar
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.25 / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.25 / 2
    
    # Align to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d Chop filter: CHOP > 61.8 = choppy (range) market, good for mean reversion/breakouts
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest high - lowest low over period))
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr1.iloc[0] = 0  # first bar has no previous close
    tr2.iloc[0] = np.abs(high_1d[0] - close_1d[0])
    tr3.iloc[0] = np.abs(low_1d[0] - close_1d[0])
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / np.log10(highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d EMA34 warmup and volume MA
    start_idx = max(50, 21)  # EMA34 needs ~34, plus buffers
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_choppy = chop_aligned[i] > 61.8  # choppy/range market
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + trend + volume + chop filter
            # Long: price breaks above camarilla H3 AND bullish bias AND volume spike AND choppy market
            long_entry = (curr_high > camarilla_h3_aligned[i]) and bullish_bias and vol_spike and is_choppy
            # Short: price breaks below camarilla L3 AND bearish bias AND volume spike AND choppy market
            short_entry = (curr_low < camarilla_l3_aligned[i]) and bearish_bias and vol_spike and is_choppy
            
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
            # Exit: price falls below camarilla L3 (invalidates breakout) OR loss of bullish bias
            if (curr_low < camarilla_l3_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above camarilla H3 (invalidates breakout) OR loss of bearish bias
            if (curr_high > camarilla_h3_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0