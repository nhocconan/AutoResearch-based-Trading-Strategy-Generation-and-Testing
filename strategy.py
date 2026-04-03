#!/usr/bin/env python3
"""
Experiment #1859: 6h Elder Ray + Regime Filter (ADX) + Volume Spike
HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure. Combined with ADX regime filter (ADX>25 = trending) and volume confirmation (>2x average), this captures strong trending moves while avoiding choppy markets. Works in both bull and bear markets by following the dominant 1d trend. Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing of 0.25 to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1859_6h_elder_ray_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Elder Ray (Bull Power / Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA(50) and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or extended adverse move ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if Bear Power becomes strongly negative (selling pressure)
                if bear_power[i] < -np.std(bear_power[max(0, i-50):i]) * 1.5:
                    exit_signal = True
                # Exit if ADX weakens (trend ending)
                elif adx[i] < 20:
                    exit_signal = True
                # Exit if 1d trend flips
                elif trend_1d_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Exit if Bull Power becomes strongly positive (buying pressure)
                if bull_power[i] > np.std(bull_power[max(0, i-50):i]) * 1.5:
                    exit_signal = True
                # Exit if ADX weakens (trend ending)
                elif adx[i] < 20:
                    exit_signal = True
                # Exit if 1d trend flips
                elif trend_1d_aligned[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Require trending market (ADX > 25)
        trending = adx[i] > 25
        
        # Volume confirmation: require volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if trending and volume_spike:
            # Elder Ray signals: Bull Power > 0 and Bear Power < 0 indicates bullish pressure
            # For long: Bull Power positive AND Bear Power not too negative
            # For short: Bear Power negative AND Bull Power not too positive
            if trend_bias > 0 and bull_power[i] > 0 and bear_power[i] < ema_13[i] * 0.001:  # Near zero bear power
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif trend_bias < 0 and bear_power[i] < 0 and bull_power[i] > ema_13[i] * 0.001:  # Near zero bull power
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals