#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
    # Long when price breaks above H3 (bullish bias) + 1d volume > 1.5x average + chop < 61.8 (trending)
    # Short when price breaks below L3 (bearish bias) + 1d volume > 1.5x average + chop < 61.8 (trending)
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year)
    # Camarilla levels provide institutional support/resistance; volume confirms breakout strength
    # Chop filter avoids false breakouts in ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla calculation, volume, and chop (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First value invalid due to roll
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Resistance levels
    H3 = pivot + (range_val * 1.1 / 4)
    H4 = pivot + (range_val * 1.1 / 2)
    # Support levels
    L3 = pivot - (range_val * 1.1 / 4)
    L4 = pivot - (range_val * 1.1 / 2)
    
    # Align 1d Camarilla levels to 12h (wait for completed 1d bar)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d Choppiness Index (14-period) for regime filter
    # Chop = 100 * log10(sum(ATR) / (log10(n) * (highest_high - lowest_low))) / log10(n)
    tr1 = np.abs(np.roll(high_1d, 1) - low_1d)  # |high_prev - low_curr|
    tr2 = np.abs(np.roll(low_1d, 1) - high_1d)  # |low_prev - high_curr|
    tr3 = np.abs(high_1d - low_1d)              # |high_curr - low_curr|
    tr1[0] = np.nan  # First value invalid due to roll
    tr2[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero and invalid values
    denominator = lowest_low - highest_high
    chop_raw = np.where(
        (denominator != 0) & (~np.isnan(denominator)) & (highest_high != lowest_low),
        100 * np.log10(np.sum(atr) / (np.log10(14) * np.abs(denominator))) / np.log10(14),
        50  # Default to neutral when invalid
    )
    # For rolling sum of ATR, we need to compute it properly
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where(
        (denominator != 0) & (~np.isnan(denominator)) & (highest_high != lowest_low) & (~np.isnan(atr_sum)),
        100 * np.log10(atr_sum / (np.log10(14) * np.abs(denominator))) / np.log10(14),
        50  # Default to neutral when invalid
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period average
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_confirm = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Chop regime filter: trending market (chop < 61.8)
        chop_value = chop_aligned[i]
        trending_regime = chop_value < 61.8
        
        # Breakout conditions
        bullish_breakout = close[i] > H3_aligned[i] and volume_confirm and trending_regime
        bearish_breakout = close[i] < L3_aligned[i] and volume_confirm and trending_regime
        
        # Exit conditions: price returns to pivot level or opposite breakout
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        long_exit = close[i] < pivot_aligned[i] or bearish_breakout
        short_exit = close[i] > pivot_aligned[i] or bullish_breakout
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
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

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0