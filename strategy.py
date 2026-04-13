#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d trend regime (ADX) + volume confirmation.
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    # Long when Bull Power > 0 + ADX_1d > 25 (trending) + volume > 1.5x 20-period average
    # Short when Bear Power > 0 + ADX_1d > 25 + volume > 1.5x 20-period average
    # Exit when power crosses zero (Bull Power <= 0 for long exit, Bear Power <= 0 for short exit)
    # Elder Ray measures trend strength via price relative to EMA; ADX filters for trending markets only.
    # Works in both bull (strong longs) and bear (strong shorts) regimes.
    # Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA(13) on 6h for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # >0 indicates bulls in control
    bear_power = ema_13 - low   # >0 indicates bears in control
    
    # Get 1d data for ADX trend regime and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate True Range (TR) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate +DM and -DM on 1d
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing function (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        alpha = 1.0 / period
        smoothed = np.zeros_like(values)
        if len(values) >= period:
            smoothed[period-1] = np.mean(values[:period])  # First value is simple average
            for i in range(period, len(values)):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / np.maximum(atr_1d, 1e-10)
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / np.maximum(atr_1d, 1e-10)
    
    # Calculate ADX (14) on 1d
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / np.maximum(plus_di_1d + minus_di_1d, 1e-10)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Calculate volume moving average (20-period) on 1d
    vol_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX_1d > 25 indicates trending market
        regime_filter = adx_1d_aligned[i] > 25
        
        # Volume confirmation: current 1d volume > 1.5x 20-period EMA
        volume_spike = volume_1d_aligned[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Elder Ray signals
        long_signal = bull_power[i] > 0 and regime_filter and volume_spike
        short_signal = bear_power[i] > 0 and regime_filter and volume_spike
        
        # Exit conditions: power crosses zero
        long_exit = bull_power[i] <= 0
        short_exit = bear_power[i] <= 0
        
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

name = "6h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0