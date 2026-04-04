#!/usr/bin/env python3
"""
Experiment #2791: 6h Elder Ray Power + 1d Regime Filter (Bull/Bear/Range)
HYPOTHESIS: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures
buying/selling pressure strength. Combined with 1d regime (ADX for trend, BB Width for
chop), we enter long when Bull Power > 0 in bull trend, short when Bear Power > 0 in
bear trend, and mean-revert at extremes in ranging markets. 6h timeframe balances
trade frequency and signal quality. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2791_6h_elder_ray_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime and EMA13 ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 1d Indicators for Regime: ADX(14) and Bollinger Bands Width ===
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ , DM-
    tr_period = 14
    tr_sum = pd.Series(tr).ewm(alpha=1/tr_period, min_periods=tr_period, adjust=False).mean().values
    dm_plus_sum = pd.Series(dm_plus).ewm(alpha=1/tr_period, min_periods=tr_period, adjust=False).mean().values
    dm_minus_sum = pd.Series(dm_minus).ewm(alpha=1/tr_period, min_periods=tr_period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, min_periods=tr_period, adjust=False).mean().values
    
    # Bollinger Bands Width
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + bb_std * std_bb
    lower_bb = sma_bb - bb_std * std_bb
    bb_width = (upper_bb - lower_bb) / sma_bb
    
    # Align regime indicators
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # === 6h Indicators: EMA13 for Elder Ray (using 6h close) ===
    ema13_6h = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power and Bear Power on 6h
    bull_power_6h = high - ema13_6h
    bear_power_6h = ema13_6h - low
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = 100  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(bb_width_aligned[i]) or
            np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Classification ---
        # Trend: ADX > 25
        # Chop: BB Width < 0.05 (low volatility) OR ADX < 20
        is_trend = adx_aligned[i] > 25
        is_chop = bb_width_aligned[i] < 0.05 or adx_aligned[i] < 20
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long
                # Exit conditions
                if is_trend:
                    # In trend: exit when power fades
                    if bull_power_6h[i] <= 0:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Chop regime
                    # In chop: exit when power reverses or at opposite extreme
                    if bear_power_6h[i] > 0:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
            else:  # Short
                if is_trend:
                    if bear_power_6h[i] <= 0:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
                else:
                    if bull_power_6h[i] > 0:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        if is_trend:
            # Trend following: follow the power
            if bull_power_aligned[i] > 0 and bear_power_aligned[i] < bull_power_aligned[i]:
                # Strong bull power, weak bear power -> long
                in_position = True
                position_side = 1
                signals[i] = SIZE
            elif bear_power_aligned[i] > 0 and bull_power_aligned[i] < bear_power_aligned[i]:
                # Strong bear power, weak bull power -> short
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Mean reversion in chop: fade extremes
            # Normalize power values for comparison
            power_ratio = bull_power_aligned[i] / (bull_power_aligned[i] + bear_power_aligned[i] + 1e-10)
            if power_ratio > 0.7:  # Extremely bullish -> short (fade)
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            elif power_ratio < 0.3:  # Extremely bearish -> long (fade)
                in_position = True
                position_side = 1
                signals[i] = SIZE
            else:
                signals[i] = 0.0
    
    return signals