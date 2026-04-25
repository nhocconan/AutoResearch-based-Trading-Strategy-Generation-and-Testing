#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian breakouts capture strong momentum, 1d EMA34 filters primary trend,
volume spike confirms institutional participation, and chop filter avoids extreme
choppy or trending markets. Works in bull/bear via trend filter and regime adaptation.
Target: 20-50 trades/year on 4h.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) channels from previous 20 periods
    # Using rolling window on high/low for breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index filter (avoid ranging AND extreme trending markets)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    # Simplified: we avoid both chop > 61.8 (ranging) and chop < 38.2 (strong trend)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    price_range_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values - pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    price_range_14 = np.where(price_range_14 == 0, 1e-10, price_range_14)
    chop = 100 * np.log10(atr_14 * 14 / price_range_14) / np.log10(14)
    chop_filter = (chop > 38.2) & (chop < 61.8)  # Only trade in moderate chop regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34, 20, 20, 14) + 1  # +1 for previous bar reference
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        
        # Donchian breakout conditions
        breakout_long = curr_high > donchian_high[i]  # Break above upper band
        breakout_short = curr_low < donchian_low[i]   # Break below lower band
        
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + trend alignment + volume + chop filter
            long_entry = breakout_long and uptrend and vol_spike and chop_ok
            short_entry = breakout_short and downtrend and vol_spike and chop_ok
            
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
            # Exit: price retouches lower Donchian band OR trend reverses
            if curr_close < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price retouches upper Donchian band OR trend reverses
            if curr_close > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0