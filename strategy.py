#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout from prior 1d with 1d volume spike and ADX trend filter
# - Long: price breaks above Camarilla H3 level (from prior 1d) + 1d volume > 2.0x 20-period MA + ADX(14) > 25
# - Short: price breaks below Camarilla L3 level (from prior 1d) + 1d volume > 2.0x 20-period MA + ADX(14) > 25
# - Exit: price returns to Camarilla Pivot level (from prior 1d) or ATR-based stoploss (2.5x ATR)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Using 1d HTF for Camarilla levels provides intraday support/resistance structure
# - Volume confirmation ensures institutional participation, ADX filter avoids ranging markets
# - Works in bull/bear: breakouts with trend in bull, mean reversion exits in bear ranges

name = "12h_1d_camarilla_breakout_volume_adx_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for 1d (prior completed day)
    # Pivot = (H + L + C) / 3
    # H3 = C + (H - L) * 1.1 / 4
    # L3 = C - (H - L) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = close_1d + range_1d * 1.1 / 4.0
    l3_1d = close_1d - range_1d * 1.1 / 4.0
    
    # Align Camarilla levels to 12h (using prior completed 1d bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate ADX(14) for 1d
    # ADX = 100 * smoothed(|+DI - -DI| / (+DI + -DI))
    # +DI = 100 * smoothed(+DM / TR)
    # -DI = 100 * smoothed(-DM / TR)
    # +DM = max(0, high - prev_high) if > max(0, prev_low - low) else 0
    # -DM = max(0, prev_low - low) if > max(0, high - prev_high) else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    # Calculate +DM and -DM
    high_diff = high_1d - np.roll(high_1d, 1)
    low_diff = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Calculate True Range
    high_low = high_1d - low_1d
    high_prev_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_prev_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    tr[0] = high_low[0]  # First bar TR is just high-low
    
    # Calculate smoothed +DM, -DM, and TR using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        """Calculate Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        return pd.Series(values).ewm(alpha=1/period, adjust=False).mean().values
    
    period = 14
    smoothed_plus_dm = wilders_smoothing(plus_dm, period)
    smoothed_minus_dm = wilders_smoothing(minus_dm, period)
    smoothed_tr = wilders_smoothing(tr, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero when both DI are zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate ATR(14) for 12h stoploss
    high_diff_12h = high_12h - np.roll(high_12h, 1)
    low_diff_12h = np.roll(low_12h, 1) - low_12h
    high_low_12h = high_12h - low_12h
    high_prev_close_12h = np.abs(high_12h - np.roll(close_12h, 1))
    low_prev_close_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(high_low_12h, np.maximum(high_prev_close_12h, low_prev_close_12h))
    tr_12h[0] = high_low_12h[0]
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h OHLC
        close_price = close_12h[i]
        high_price = high_12h[i]
        low_price = low_12h[i]
        
        # Get aligned 1d data for current 12h bar (completed 1d bar)
        pivot_current = pivot_aligned[i]
        h3_current = h3_aligned[i]
        l3_current = l3_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_12h_current = align_htf_to_ltf(prices, df_1d, volume_12h)[i]
        adx_current = adx_aligned[i]
        
        # Volume spike condition: current 12h volume > 2.0x 20-period MA (from 1d)
        volume_spike = volume_12h_current > 2.0 * volume_ma_current
        
        # Trend condition: ADX > 25
        trending = adx_current > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above H3 + volume spike + trending
            if (close_price > h3_current and volume_spike and trending):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below L3 + volume spike + trending
            elif (close_price < l3_current and volume_spike and trending):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to pivot level
            # 2. ATR-based stoploss (2.5x ATR from entry)
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit if price returns to pivot
                if close_price <= pivot_current:
                    exit_signal = True
                # Exit if stoploss hit (2.5x ATR below entry)
                elif low_price <= entry_price - 2.5 * atr_12h[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price returns to pivot
                if close_price >= pivot_current:
                    exit_signal = True
                # Exit if stoploss hit (2.5x ATR above entry)
                elif high_price >= entry_price + 2.5 * atr_12h[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals