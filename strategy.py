#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and choppy market filter (Choppiness Index > 61.8) captures strong institutional moves in ranging markets while avoiding whipsaws in strong trends. Designed for 20-40 trades/year to minimize fee drag. Works in both bull and bear markets via trend filter and regime adaptation.
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
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Use previous completed 1d bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) for regime filter
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index"""
        atr = np.zeros_like(close_arr)
        tr1 = np.abs(high_arr - low_arr)
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Calculate ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr[period-1] = np.mean(tr[1:period])  # Seed with simple average
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close_arr)
        ll = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            hh[i] = np.max(high_arr[i-period+1:i+1])
            ll[i] = np.min(low_arr[i-period+1:i+1])
        
        # Choppiness Index formula
        chop = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            if hh[i] - ll[i] != 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50  # Neutral when no range
        return chop
    
    chop_values = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 34)  # EMA34, vol MA, chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(chop_values[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_34_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        chop_val = chop_values[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # Choppiness regime filter: only trade in choppy markets (CHOP > 61.8)
        choppy_market = chop_val > 61.8
        
        if position == 0:
            # Look for entry signals: Camarilla R1/S1 breakout with trend and volume in choppy market
            # Long: price breaks above R1 with uptrend (close > EMA34) and volume spike in choppy market
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike and choppy_market
            # Short: price breaks below S1 with downtrend (close < EMA34) and volume spike in choppy market
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike and choppy_market
            
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
            # 1. Opposite breakout: price breaks below S1 (exit long)
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Regime change: exit if market becomes trending (CHOP < 38.2)
            elif chop_val < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R1 (exit short)
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Regime change: exit if market becomes trending (CHOP < 38.2)
            elif chop_val < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0