#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike (>1.8x 20-bar average), and chop regime filter (CHOP<61.8 for trending). Designed to capture strong momentum bursts in both bull and bear markets by requiring confluence of price level break, trend alignment, volume confirmation, and trending regime. Targets 12-37 trades/year on 12h to minimize fee drag while maintaining edge in volatile regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, EMA trend, and chop regime - HTF timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (R1, S1) on 1d using previous bar's close to avoid look-ahead
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Use previous bar's data (shift by 1) to ensure we only use completed 1d bar
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Shift by 1 to use only completed 1d bar
    camarilla_r1 = np.concatenate([[np.nan], camarilla_r1[:-1]])
    camarilla_s1 = np.concatenate([[np.nan], camarilla_s1[:-1]])
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Choppiness Index (CHOP) on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    # Simplified: CHOP = 100 * log10(sum(True Range over period) / (log10(period) * (max(high) - min(low))))
    # We'll use a rolling window of 14 periods
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Sum of True Range over window
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Max(high) - Min(low) over window
        max_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        range_hl = max_high - min_low
        
        # Avoid division by zero
        range_hl = np.where(range_hl == 0, np.nan, range_hl)
        
        # Choppiness Index
        chop = 100 * np.log10(tr_sum / (np.log10(window) * range_hl))
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume average (20-period) for volume spike filter on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 34, 14, 20)  # vol needs 20, EMA needs 34, CHOP needs 14, vol_ma needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema_val = ema_34_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 1.8x 20-period average
        volume_spike = vol_val > 1.8 * vol_ma_val
        
        # Choppiness regime filter: CHOP < 61.8 indicates trending regime (favor breakouts)
        trending_regime = chop_val < 61.8
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend, volume, and regime confirmation
            # Long: price breaks above R1 with EMA uptrend, volume spike, and trending regime
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike and trending_regime
            # Short: price breaks below S1 with EMA downtrend, volume spike, and trending regime
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike and trending_regime
            
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
            # 1. Stoploss: price moves below S1 (opposite Camarilla level)
            if low_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes below EMA34
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves above R1 (opposite Camarilla level)
            if high_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes above EMA34
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0