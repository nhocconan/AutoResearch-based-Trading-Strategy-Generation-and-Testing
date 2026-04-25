#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + 1d EMA50 Trend + Volume Spike + Chop Filter (Regime Adaptive)
Hypothesis: Camarilla H3/L3 breakouts capture institutional intraday extremes, 
1d EMA50 filters primary trend, volume spike confirms participation, 
and chop filter adapts to regime (trend when CHOP<50, mean-revert when CHOP>61.8).
Works in bull/bear via trend filter and regime-specific logic.
Target: 20-40 trades/year on 4h.
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
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla H3, L3 levels (H3 = Close + 1.1*(High-Low)/6, L3 = Close - 1.1*(High-Low)/6)
    camarilla_h3 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    camarilla_l3 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index filter (regime detection)
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 14-period high-low range
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14)/range14) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        # Regime detection
        is_trending = chop_val < 50.0
        is_choppy = chop_val > 61.8
        
        # Camarilla breakout conditions
        breakout_long = curr_high > camarilla_h3_aligned[i]
        breakout_short = curr_low < camarilla_l3_aligned[i]
        
        # Trend filter
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-adaptive entry logic
            if is_trending:
                # Trending market: breakout with trend
                long_entry = breakout_long and uptrend and vol_spike
                short_entry = breakout_short and downtrend and vol_spike
            elif is_choppy:
                # Choppy market: mean-reversion at extremes
                long_entry = breakout_short and not downtrend and vol_spike  # Break below L3 but expect reversal up
                short_entry = breakout_long and not uptrend and vol_spike   # Break above H3 but expect reversal down
            else:
                # Transition regime: require both volume and trend alignment
                long_entry = breakout_long and uptrend and vol_spike
                short_entry = breakout_short and downtrend and vol_spike
            
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
            # Exit: trend reversal OR price reaches opposite Camarilla level
            if not uptrend or curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: trend reversal OR price reaches opposite Camarilla level
            if not downtrend or curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike_ChopFilter_RegimeAdaptive"
timeframe = "4h"
leverage = 1.0