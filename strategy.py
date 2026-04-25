#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with 1d EMA34 Trend Filter and Volume Spike + Choppiness Filter
Hypothesis: Donchian(20) breakouts capture strong momentum. Adding 1d EMA34 trend filter (more stable than 12h) and volume spike confirmation reduces false breakouts.
Choppiness filter avoids whipsaws in ranging markets. Targets 20-50 trades/year by requiring confluence of four conditions.
Works in bull (long breakouts above EMA34) and bear (short breakouts below EMA34) regimes.
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
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness filter: avoid trading when CHOP > 61.8 (ranging market)
    # CHOP = 100 * log10(sum(ATR(1), n) / (log10(n) * (max(high,n) - min(low,n))))
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(np.absolute(low - np.roll(close, 1)), tr1)
    tr2[0] = high[0] - low[0]  # first bar TR
    atr1 = pd.Series(tr2).rolling(window=1, min_periods=1).sum().values
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (np.log10(14) * (max_high - min_low)))
    chop_filter = chop < 61.8  # only trade when NOT choppy (trending market)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20), EMA34, and CHOP(14)
    start_idx = max(20, 34, 14)  # Donchian lookback, EMA34, CHOP period
    
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
        not_choppy = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Donchian breakout + trend + volume + not choppy
            # Long: price breaks above Donchian upper AND bullish bias AND volume spike AND not choppy
            long_entry = (curr_high > donchian_upper[i]) and bullish_bias and vol_spike and not_choppy
            # Short: price breaks below Donchian lower AND bearish bias AND volume spike AND not choppy
            short_entry = (curr_low < donchian_lower[i]) and bearish_bias and vol_spike and not_choppy
            
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
            # Exit: price falls below Donchian lower (mean reversion) OR loss of bullish bias OR choppy market
            if (curr_low < donchian_lower[i]) or (curr_close < ema_1d_aligned[i]) or (not not_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper (mean reversion) OR loss of bearish bias OR choppy market
            if (curr_high > donchian_upper[i]) or (curr_close > ema_1d_aligned[i]) or (not not_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0