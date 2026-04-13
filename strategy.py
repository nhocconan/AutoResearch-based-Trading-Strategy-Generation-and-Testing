#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla Pivot Breakout + 12h Volume Spike + 1d Chop Regime Filter
    # Long: Price breaks above Camarilla R4 (12h) AND 12h volume > 1.5x 20-period avg AND 1d chop < 61.8 (trending)
    # Short: Price breaks below Camarilla S4 (12h) AND 12h volume > 1.5x 20-period avg AND 1d chop < 61.8 (trending)
    # Exit: Price retreats to Camarilla PP (12h) OR chop > 61.8 (ranging)
    # Uses 12h for Camarilla pivots (structure), 12h for volume confirmation, 1d for chop regime (trend vs range)
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and volume (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for chop regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    # Camarilla: PP = (H+L+C)/3, R4 = C + ((H-L)*1.1/2), S4 = C - ((H-L)*1.1/2)
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Previous bar values for Camarilla calculation (no look-ahead)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    # First bar will have NaN due to roll, handled later
    
    # Camarilla calculations
    pp_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    r4_12h = prev_close_12h + ((prev_high_12h - prev_low_12h) * 1.1 / 2.0)
    s4_12h = prev_close_12h - ((prev_high_12h - prev_low_12h) * 1.1 / 2.0)
    
    # Align 12h Camarilla levels to 6h (wait for completed 12h bar)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate 12h volume and 20-period average (for volume spike)
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_12h / vol_ma_20  # Current volume / 20-period average
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio)
    
    # Calculate 1d Chop Index (Ehler's Chop: measures trend vs ranging)
    # Chop = 100 * log10(sum(ATR1) / (n * (HHV - LLV))) / log10(n)
    # Simplified: Chop > 61.8 = ranging, Chop < 38.2 = strong trend
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR14 (using Wilder's smoothing)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Sum of ATR1 over 14 periods (already smoothed)
    sum_atr = atr_1d  # ATR14 is already the smoothed average true range
    
    # 14-period highest high and lowest low
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    numerator = np.nansum(sum_atr)  # This needs to be rolling sum - fix below
    # Recalculate properly: rolling sum of ATR14 over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = hh_1d - ll_1d
    
    # Avoid division by zero
    chop_raw = np.full_like(close_1d, np.nan)
    valid_mask = (~np.isnan(sum_atr_14)) & (~np.isnan(hh_ll_diff)) & (hh_ll_diff > 0)
    chop_raw[valid_mask] = 100 * np.log10(sum_atr_14[valid_mask] / hh_ll_diff[valid_mask]) / np.log10(14)
    
    # Align 1d Chop to 6h (wait for completed 1d bar)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d chop < 61.8 (trending market)
        trending_regime = chop_aligned[i] < 61.8
        # Exit regime: chop > 61.8 (ranging market)
        ranging_regime = chop_aligned[i] > 61.8
        
        # Volume confirmation: volume spike > 1.5x average
        volume_spike = vol_ratio_aligned[i] > 1.5
        
        # Breakout conditions
        long_breakout = close[i] > r4_aligned[i]
        short_breakout = close[i] < s4_aligned[i]
        
        # Entry logic: Breakout + volume spike + trending regime
        long_entry = long_breakout and volume_spike and trending_regime
        short_entry = short_breakout and volume_spike and trending_regime
        
        # Exit logic: Price retreats to PP OR regime shifts to ranging
        long_exit = (close[i] < pp_aligned[i]) or ranging_regime
        short_exit = (close[i] > pp_aligned[i]) or ranging_regime
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "6h"
leverage = 1.0