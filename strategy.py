#!/usr/bin/env python3
"""
Experiment #3839: 6h Elder Ray + 12h ADX + Volume Spike
HYPOTHESIS: Elder Ray (bull/bear power) identifies trend strength, 12h ADX filters for trending regimes (>25), and 6h volume spikes (>2.0x) confirm institutional participation. Works in bull markets (bull power + ADX) and bear markets (bear power + ADX). Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3839_6h_elder_ray_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX
    period_adx = 14
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            first_avg = np.nansum(data[1:period+1]) / period
            result[period] = first_avg
            for i in range(period+1, len(data)):
                if not np.isnan(data[i]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilder_smooth(tr, period_adx)
    dm_plus_smooth = wilder_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilder_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period_adx)
    
    # Align 12h ADX to 6h timeframe (shifted by 1 for completed 12h bar)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    period_ema = 13
    ema_close = pd.Series(close).ewm(span=period_ema, min_periods=period_ema, adjust=False).mean().values
    bull_power = high - ema_close
    bear_power = low - ema_close
    
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(period_ema + period_adx, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                # Calculate ATR manually for exit condition
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price < highest_since_entry - 2.5 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if bear power becomes strongly negative (trend weakness)
                    elif bear_power[i] < -1.5 * np.std(bear_power[max(0, i-50):i+1]):
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price > lowest_since_entry + 2.5 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if bull power becomes strongly positive (trend weakness)
                    elif bull_power[i] > 1.5 * np.std(bull_power[max(0, i-50):i+1]):
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) and ADX > 25 (trending market)
        volume_spike = vol_ratio[i] > 2.0
        strong_trend = adx_12h_aligned[i] > 25
        
        if volume_spike and strong_trend:
            # Long entry: Bull power positive and increasing (bullish momentum)
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Bear power negative and decreasing (bearish momentum)
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals