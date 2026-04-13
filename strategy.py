#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R mean reversion with 1d volume spike filter and 1w trend regime.
    # Long when Williams %R(14) < -80 (oversold) + volume spike (>2.0x 20-period avg) + ADX_1w > 25 (trending).
    # Short when Williams %R(14) > -20 (overbought) + volume spike + ADX_1w > 25.
    # Exit when Williams %R crosses back above -50 (long) or below -50 (short).
    # Uses Williams %R for mean reversion in trends, volume spike for confirmation, ADX to ensure we're in a trend.
    # Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14) on 12h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / np.maximum(highest_high - lowest_low, 1e-10)
    
    # Get 1d data for volume spike filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate volume moving average (20-period) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_spike = volume_1d_aligned[i] > 2.0 * vol_ma_1d_aligned[i]
        
        # Regime filter: ADX_1w > 25 indicates trending market (good for mean reversion pullbacks)
        regime_filter = adx_1w_aligned[i] > 25
        
        # Mean reversion conditions
        long_signal = williams_r[i] < -80 and volume_spike and regime_filter
        short_signal = williams_r[i] > -20 and volume_spike and regime_filter
        
        # Exit conditions: Williams %R crosses back above -50 (long) or below -50 (short)
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
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

name = "12h_1d_1w_williamsr_volume_adx_v1"
timeframe = "12h"
leverage = 1.0