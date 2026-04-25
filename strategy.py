#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with 1d EMA34 Trend Filter and Volume Spike + Chop Filter
Hypothesis: Donchian(20) breakouts capture strong momentum. When aligned with 1d EMA34 trend,
confirmed by volume spikes, and filtered by choppiness regime (CHOP > 61.8 for mean reversion),
this strategy works in both bull (long breakouts in trending markets) and bear (short breakouts).
Adding chop filter reduces false breakouts in ranging markets, improving trade quality and reducing overtrading.
Designed for 4h timeframe to target 20-50 trades/year (75-200 over 4 years) by requiring confluence of
Donchian breakout, 1d EMA34 trend, volume confirmation, and chop regime filter.
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian(20) channels on primary timeframe (4h)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    # We use the previous 20 bars (excluding current) to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index (CHOP) filter - use 14-period
    # CHOP > 61.8 = ranging market (good for mean reversion breakouts)
    # CHOP < 38.2 = trending market (good for trend following)
    # We'll use CHOP > 50 as a filter to avoid strong trends where breakouts may fail
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_ll = hh - ll
    chop = np.zeros_like(hh_ll)
    mask = (hh_ll > 0) & (tr_sum > 0)
    chop[mask] = 100 * np.log10(tr_sum[mask] / hh_ll[mask]) / np.log10(14)
    # For invalid values, set to 50 (neutral)
    chop[~mask] = 50.0
    
    # Chop filter: we want ranging markets for mean reversion breakouts
    # CHOP > 50 indicates more ranging/choppy behavior
    chop_filter = chop > 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20), EMA34, and CHOP(14)
    start_idx = max(20, 34, 14)  # Donchian lookback, EMA34, CHOP
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_val = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Donchian breakout + trend + volume + chop filter
            # Long: price breaks above Donchian upper AND bullish bias AND volume spike AND chop filter
            long_entry = (curr_high > donchian_upper[i]) and bullish_bias and vol_spike and chop_val
            # Short: price breaks below Donchian lower AND bearish bias AND volume spike AND chop filter
            short_entry = (curr_low < donchian_lower[i]) and bearish_bias and vol_spike and chop_val
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian lower (mean reversion) OR loss of bullish bias
            if (curr_low < donchian_lower[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper (mean reversion) OR loss of bearish bias
            if (curr_high > donchian_upper[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0