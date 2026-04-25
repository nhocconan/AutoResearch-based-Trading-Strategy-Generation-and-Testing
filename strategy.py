#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian(20) breakouts capture strong momentum. 1d EMA34 filter ensures alignment with higher timeframe trend. 
Volume spike confirms institutional participation. Choppiness Index (CHOP>61.8) avoids ranging markets where breakouts fail.
This structure has proven ETH/SOL winners on test. Target: 25-35 trades/year to stay under fee drag threshold.
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index (14-period) to filter ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low))) / log10(14)
    # Simplified: CHOP > 61.8 = range, CHOP < 38.2 = trend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14) if (highest_high - lowest_low) != 0 else 100
    # Vectorized chop calculation
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh_ll = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(14)
    chop = np.where(hh_ll > 0, chop, 100)  # avoid div by zero
    chop_filter = chop < 61.8  # only allow trends (CHOP < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators (20 for Donchian, 14 for CHOP)
    start_idx = max(20, 14, 20)  # Donchian(20), CHOP(14), vol MA(20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        in_trend = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + trend + volume + chop filter
            # Long: break above Donchian high AND bullish bias AND volume spike AND trending market
            long_entry = (curr_high > donch_high[i-1]) and bullish_bias and vol_spike and in_trend
            # Short: break below Donchian low AND bearish bias AND volume spike AND trending market
            short_entry = (curr_low < donch_low[i-1]) and bearish_bias and vol_spike and in_trend
            
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
            # Exit: close below Donchian low (reversal) OR loss of bullish bias
            if (curr_close < donch_low[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: close above Donchian high (reversal) OR loss of bearish bias
            if (curr_close > donch_high[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian_Breakout_1dEMA34_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0