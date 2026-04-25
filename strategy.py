#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels on 1d identify key daily support/resistance; breakouts with 1d EMA34 trend filter, volume confirmation, and chop regime filter capture momentum swings while avoiding false breakouts in sideways markets. Designed for 4h timeframe to target 20-50 trades/year (75-200 over 4 years), minimizing fee drag. Works in both bull and bear markets by following the 1d trend and avoiding counter-trend entries.
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d
    # Typical price = (H+L+C)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla width = (H - L) * 1.1 / 12
    camarilla_width = (df_1d['high'] - df_1d['low']) * 1.1 / 12
    # H3 = C + width * 1.1/4, L3 = C - width * 1.1/4
    h3 = typical_price + camarilla_width * 1.1 / 4
    l3 = typical_price - camarilla_width * 1.1 / 4
    # H4 = C + width * 1.1/2, L4 = C - width * 1.1/2 (stronger breakout levels)
    h4 = typical_price + camarilla_width * 1.1 / 2
    l4 = typical_price - camarilla_width * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4.values)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Chopiness Index filter (14-period) - avoid trading in choppy markets
    # Chop = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # Simplified: use rolling ATR ratio
    atr_period = 14
    tr1 = np.abs(high - np.roll(low, 1))
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    # Chop approximation: high-low range relative to ATR
    hl_range = high - low
    chop_value = 100 * np.log10(hl_range / (atr * atr_period)) / np.log10(atr_period)
    # Normalize chop to 0-100 scale (simplified)
    chop_value = np.where(atr > 0, 100 * (hl_range / (atr * atr_period)), 50)
    chop_value = np.clip(chop_value, 0, 100)
    chop_filter = chop_value > 38.2  # Only trade when not too choppy (trending regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34, 14)  # volume MA, EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_trending = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3/H4 AND bullish bias AND volume spike AND trending market
            long_entry = ((curr_high > h3_aligned[i]) or (curr_high > h4_aligned[i])) and bullish_bias and vol_spike and is_trending
            # Short: price breaks below L3/L4 AND bearish bias AND volume spike AND trending market
            short_entry = ((curr_low < l3_aligned[i]) or (curr_low < l4_aligned[i])) and bearish_bias and vol_spike and is_trending
            
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
            # Exit: price falls below L3 (mean reversion) OR loss of bullish bias OR choppy market
            if (curr_low < l3_aligned[i]) or (curr_close < ema_1d_aligned[i]) or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (mean reversion) OR loss of bearish bias OR choppy market
            if (curr_high > h3_aligned[i]) or (curr_close > ema_1d_aligned[i]) or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0