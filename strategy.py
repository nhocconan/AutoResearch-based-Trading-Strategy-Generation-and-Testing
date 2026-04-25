#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + Choppiness Filter + ATR Stop
Hypothesis: Donchian(20) breakouts capture momentum, volume spike confirms institutional interest,
choppiness filter avoids whipsaws in ranging markets, ATR-based stop manages risk.
Designed for 4h timeframe with 75-200 total trades over 4 years to balance opportunity and fee drag.
Works in both bull (breakouts continue) and bear (breakdowns accelerate) markets.
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
    
    # Get daily data for choppiness regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough days for choppiness calculation
        return np.zeros(n)
    
    # Calculate Choppiness Index on daily data (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for daily
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First value
    
    # ATR 14
    atr_14_1d = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Sum of TR over 14 periods
    sum_tr_14 = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        sum_tr_14[i] = np.sum(tr_1d[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_14_1d = np.full(len(high_1d), np.nan)
    ll_14_1d = np.full(len(low_1d), np.nan)
    for i in range(14, len(high_1d)):
        hh_14_1d[i] = np.max(high_1d[i-13:i+1])
        ll_14_1d[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sumTR14 / (ATR14 * 14)) / log10(14)
    chop_1d = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        if atr_14_1d[i] > 0 and sum_tr_14[i] > 0:
            chop_1d[i] = 100 * np.log10(sum_tr_14[i] / (atr_14_1d[i] * 14)) / np.log10(14)
    
    # Align choppiness to 4h timeframe (needs 2 extra days for confirmation like fractals)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d, additional_delay_bars=2)
    
    # Calculate Donchian channels (20-period) on 4h
    donch_h = np.full(n, np.nan)
    donch_l = np.full(n, np.nan)
    for i in range(20, n):
        donch_h[i] = np.max(high[i-19:i+1])
        donch_l[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR (10-period) for stoploss
    atr = np.full(n, np.nan)
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for i in range(10, n):
        atr[i] = np.mean(tr[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Donchian, volume MA, ATR, and choppiness
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_h[i]) or np.isnan(donch_l[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donch_high = donch_h[i]
        donch_low = donch_l[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        chop_val = chop_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        # Choppiness filter: only trade when CHOP < 50 (trending market)
        trending_regime = chop_val < 50
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Donchian high with volume confirmation in trending market
            long_breakout = (curr_close > donch_high) and volume_confirm and trending_regime
            # Short: price breaks below Donchian low with volume confirmation in trending market
            short_breakout = (curr_close < donch_low) and volume_confirm and trending_regime
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.5 * atr_val
            elif short_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.5 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: price closes below ATR stop
            # Exit: price closes below Donchian low (trailing exit)
            if curr_close < atr_stop or curr_close < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                # Trail stop up: move ATR stop to breakeven + 0.5*ATR after 1*ATR profit
                profit = curr_close - entry_price
                if profit > atr_val:
                    atr_stop = max(atr_stop, entry_price + 0.5 * atr_val)
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Stoploss: price closes above ATR stop
            # Exit: price closes above Donchian high (trailing exit)
            if curr_close > atr_stop or curr_close > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                # Trail stop down: move ATR stop to breakeven - 0.5*ATR after 1*ATR profit
                profit = entry_price - curr_close
                if profit > atr_val:
                    atr_stop = min(atr_stop, entry_price - 0.5 * atr_val)
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ChopFilter_ATRStop"
timeframe = "4h"
leverage = 1.0