#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_Trend_VolumeRegime
Hypothesis: Daily Donchian(20) breakouts with 1-week EMA50 trend filter, volume confirmation (>1.5x 20-bar avg), and chop regime filter (CHOP < 50) captures strong institutional moves while avoiding choppy markets. Uses ATR(14) stoploss (2.0) and discrete sizing (0.25). Designed for 1d timeframe to minimize fees and work in both bull and bear markets via trend filter and regime avoidance. Targets 15-25 trades/year by requiring strict confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels on 1d (using previous 20 bars to avoid look-ahead)
    # Highest high of previous 20 bars
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lowest low of previous 20 bars
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR(14) on 1d for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter on 1d (CHOP < 50 = strong trending regime)
    n_chop = 14
    tr_chop = []
    for i in range(1, len(high)):
        tr_val = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr_chop.append(tr_val)
    tr_chop = np.concatenate([[np.nan], tr_chop])
    
    atr_sum = pd.Series(tr_chop).rolling(window=n_chop, min_periods=n_chop).sum().values
    max_minus_min = pd.Series(high - low).rolling(window=n_chop, min_periods=n_chop).max().values
    chop_raw = 100 * np.log10(atr_sum / (n_chop * max_minus_min)) / np.log10(n_chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14, 20, 14)  # EMA50, Donchian, ATR, vol MA, Chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr_1d[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop_raw[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get values
        ema_val = ema_50_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        atr_val = atr_1d[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        chop_val = chop_raw[i]
        
        # Regime filter: only trade in strong trending markets (CHOP < 50)
        in_strong_trend = chop_val < 50
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with trend and volume
            # Long: price breaks above upper channel with uptrend (close > EMA50) and volume spike
            long_signal = (high_val > upper_channel) and (close_val > ema_val) and volume_spike and in_strong_trend
            # Short: price breaks below lower channel with downtrend (close < EMA50) and volume spike
            short_signal = (low_val < lower_channel) and (close_val < ema_val) and volume_spike and in_strong_trend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite breakout: price breaks below lower channel (exit long)
            elif close_val < lower_channel:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite breakout: price breaks above upper channel (exit short)
            elif close_val > upper_channel:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeRegime"
timeframe = "1d"
leverage = 1.0