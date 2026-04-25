#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels from 1d timeframe act as key intraday resistance/support.
Breakouts above H3 or below L3 with volume confirmation, aligned with 1d EMA34 trend,
and only in trending regimes (Choppiness Index < 38.2) capture momentum moves.
Designed for 4h timeframe with tight entry conditions to achieve 19-50 trades/year.
Works in bull (breakouts above H3 in uptrend) and bear (breakouts below L3 in downtrend).
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
    
    # Get 1d data for Camarilla calculation and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3) from previous day
    # Camarilla: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + (1.1 * (high_1d - low_1d) / 6)
    camarilla_l3 = close_1d - (1.1 * (high_1d - low_1d) / 6)
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed for pivot levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Choppiness Index (14-period) on 1d timeframe for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    # Simplified: Chop = 100 * log10(sum(tr_range_14) / (hh_14 - ll_14)) / log10(14)
    # We'll use a proxy: high-low range based chop for 1d
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    # Avoid division by zero
    chop_1d = np.where(range_14 > 0, 
                       100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 
                       50.0)  # neutral when range is zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_filter = chop_1d_aligned < 38.2  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        camarilla_h3_level = camarilla_h3_aligned[i]
        camarilla_l3_level = camarilla_l3_aligned[i]
        is_trending = chop_filter[i]
        
        if position == 0:
            # Look for entry signals (only in trending regime)
            if is_trending:
                # Long: price breaks above camarilla H3 AND volume spike AND price > EMA (uptrend)
                long_entry = (curr_high > camarilla_h3_level) and vol_spike and (curr_close > ema_trend)
                # Short: price breaks below camarilla L3 AND volume spike AND price < EMA (downtrend)
                short_entry = (curr_low < camarilla_l3_level) and vol_spike and (curr_close < ema_trend)
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # no entries in choppy regime
        elif position == 1:
            # Long position management
            # Exit: price crosses below camarilla L3 OR price crosses below EMA (trend change)
            if (curr_low < camarilla_l3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above camarilla H3 OR price crosses above EMA (trend change)
            if (curr_high > camarilla_h3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0