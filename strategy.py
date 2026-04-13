#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w ADX trend filter.
    # Long when price breaks above H3 pivot level + volume spike (>1.8x 24-period avg) + ADX_1w > 20.
    # Short when price breaks below L3 pivot level + volume spike + ADX_1w > 20.
    # Exit when price returns to mean (Pivot point) or opposite pivot level touched.
    # Uses Camarilla levels for intraday structure, volume for confirmation, ADX to avoid chop.
    # Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 12h bar (using high/low/close of previous bar)
    # H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), H2 = close + 1.066*(high-low)
    # L3 = close - 1.25*(high-low), L2 = close - 1.066*(high-low), L4 = close - 1.5*(high-low)
    # Pivot = (high+low+close)/3
    # We calculate these for the previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    h3 = pivot + 1.25 * range_hl
    l3 = pivot - 1.25 * range_hl
    h4 = pivot + 1.5 * range_hl  # Strong breakout level
    l4 = pivot - 1.5 * range_hl
    
    # Get 1d data for volume spike filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate volume moving average (24-period) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    
    # Align HTF volume MA to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for ADX trend regime (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range (TR) on 1w
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate +DI and -DI (14) on 1w
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilders_smoothing(values, period):
        alpha = 1.0 / period
        smoothed = np.zeros_like(values)
        if len(values) >= period:
            smoothed[period-1] = np.mean(values[:period])  # First value is simple average
            for i in range(period, len(values)):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / np.maximum(atr_1w, 1e-10)
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / np.maximum(atr_1w, 1e-10)
    
    # Calculate ADX (14) on 1w
    dx = 100 * np.abs(plus_di_1w - minus_di_1w) / np.maximum(plus_di_1w + minus_di_1w, 1e-10)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align HTF ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.8x 24-period 1d average (aligned)
        volume_spike = volume[i] > 1.8 * vol_ma_1d_aligned[i]
        
        # Regime filter: ADX_1w > 20 indicates trending market (good for breakout continuation)
        regime_filter = adx_1w_aligned[i] > 20
        
        # Breakout conditions
        long_breakout = close[i] > h3[i] and volume_spike and regime_filter
        short_breakout = close[i] < l3[i] and volume_spike and regime_filter
        
        # Exit conditions: return to pivot or touch opposite level
        long_exit = close[i] < pivot[i] or close[i] > h4[i]
        short_exit = close[i] > pivot[i] or close[i] < l4[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "12h_1d_1w_camarilla_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0