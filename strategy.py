#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
    # Long: price breaks above Camarilla H3 level (1d) + volume > 1.5x 20-period 1d avg + ADX > 25 (trending)
    # Short: price breaks below Camarilla L3 level (1d) + volume > 1.5x 20-period 1d avg + ADX > 25 (trending)
    # Exit: price returns to Camarilla pivot point (mean reversion within Camarilla levels)
    # Uses 12h primary timeframe for lower frequency and minimal fee drag
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    # Camarilla pivots provide strong intraday support/resistance levels that work in all regimes
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for Camarilla pivots and volume (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros(len(df_1d))
    
    # Calculate Camarilla pivot levels on 1d data
    # Based on previous day's OHLC
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # H2 = close + 0.55 * (high - low)
    # H1 = close + 0.275 * (high - low)
    # Pivot = (high + low + close) / 3
    # L1 = close - 0.275 * (high - low)
    # L2 = close - 0.55 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    # Use previous day's data to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first value to NaN since no previous day exists
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate pivot levels
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on 1d data (14-period)
    # ADX measures trend strength regardless of direction
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * EWMA(+DM) / ATR
    # -DI = 100 * EWMA(-DM) / ATR
    # ADX = EWMA(abs(+DI - -DI) / (+DI + -DI))
    
    # Calculate True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:]))
    tr = np.concatenate([[np.nan], tr1])  # align length
    
    # Calculate +DM and -DM
    dm_plus = np.where((high_1d[1:] - np.roll(high_1d, 1)[1:]) > (np.roll(low_1d, 1)[1:] - low_1d[1:]),
                       np.maximum(high_1d[1:] - np.roll(high_1d, 1)[1:], 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1)[1:] - low_1d[1:]) > (high_1d[1:] - np.roll(high_1d, 1)[1:]),
                        np.maximum(np.roll(low_1d, 1)[1:] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate +DI and -DI
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period 1d average
        curr_vol_1d = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmed = curr_vol_1d > 1.5 * vol_avg_20_aligned[i]
        
        # ADX filter: trending market (ADX > 25)
        is_trending = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_long = (close[i] > camarilla_h3_aligned[i] and 
                        volume_confirmed and 
                        is_trending)
        breakout_short = (close[i] < camarilla_l3_aligned[i] and 
                         volume_confirmed and 
                         is_trending)
        
        # Exit conditions: return to Camarilla pivot point
        exit_long = position == 1 and close[i] <= camarilla_pivot_aligned[i]
        exit_short = position == -1 and close[i] >= camarilla_pivot_aligned[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_pivot_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0