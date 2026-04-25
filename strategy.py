#!/usr/bin/env python3
"""
4h Camarilla H4/L4 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H4/L4 levels (stronger than H3/L3) act as major support/resistance on 4h.
Breaking above H4 with volume spike in uptrend (price > EMA34) captures momentum.
Breaking below L4 with volume spike in downtrend (price < EMA34) captures short opportunities.
Choppiness filter avoids whipsaws in ranging markets. Discrete sizing (0.0, ±0.25) minimizes fee churn.
Target: 20-40 trades/year on 4h timeframe. Works in both bull and bear via trend alignment.
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
    
    # Get 1d data for Camarilla and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d (based on previous day's high/low/close)
    # H4 = Close + 1.1 * (High - Low)
    # L4 = Close - 1.1 * (High - Low)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Choppiness Index (14) to avoid ranging markets
    # CHOP = 100 * log10(sum(ATR14) / (HHV14 - LLV14)) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # only allow trades when not strongly ranging (CHOP < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, ATR, and CHOP
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        not_choppy = chop_filter[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H4 AND volume spike AND price > EMA (uptrend) AND not choppy
            long_entry = (curr_close > h4) and vol_spike and (curr_close > ema_trend) and not_choppy
            # Short: price breaks below L4 AND volume spike AND price < EMA (downtrend) AND not choppy
            short_entry = (curr_close < l4) and vol_spike and (curr_close < ema_trend) and not_choppy
            
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
            # Exit: price crosses below L4 OR price crosses below EMA
            if (curr_close < l4) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H4 OR price crosses above EMA
            if (curr_close > h4) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0