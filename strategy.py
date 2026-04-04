#!/usr/bin/env python3
"""
Experiment #5019: 6h Donchian(20) Breakout + 12h ADX Trend Filter + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts in direction of 12h ADX>25 trend with volume confirmation (>1.5x average) capture strong momentum moves in both bull and bear markets. Uses ATR(14) trailing stop (2.0x) to limit downside. Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5019_6h_donchian20_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: ADX(14) for trend filter ===
    if len(df_12h) >= 30:  # Need enough data for ADX calculation
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # True Range
        tr1 = high_12h[1:] - low_12h[1:]
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                           np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
        dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                            np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/14)
        alpha = 1.0 / 14
        atr_12h = np.full_like(tr_12h, np.nan, dtype=np.float64)
        dm_plus_smooth = np.full_like(dm_plus, np.nan, dtype=np.float64)
        dm_minus_smooth = np.full_like(dm_minus, np.nan, dtype=np.float64)
        
        # Initialize first values
        if len(tr_12h) >= 14:
            atr_12h[13] = np.nanmean(tr_12h[1:15])
            dm_plus_smooth[13] = np.nanmean(dm_plus[1:15])
            dm_minus_smooth[13] = np.nanmean(dm_minus[1:15])
            
            # Wilder's smoothing
            for i in range(14, len(tr_12h)):
                atr_12h[i] = atr_12h[i-1] * (1 - alpha) + alpha * tr_12h[i]
                dm_plus_smooth[i] = dm_plus_smooth[i-1] * (1 - alpha) + alpha * dm_plus[i]
                dm_minus_smooth[i] = dm_minus_smooth[i-1] * (1 - alpha) + alpha * dm_minus[i]
        
        # DI+ and DI-
        di_plus = np.full_like(atr_12h, np.nan, dtype=np.float64)
        di_minus = np.full_like(atr_12h, np.nan, dtype=np.float64)
        valid_atr = ~np.isnan(atr_12h) & (atr_12h > 0)
        di_plus[valid_atr] = 100 * dm_plus_smooth[valid_atr] / atr_12h[valid_atr]
        di_minus[valid_atr] = 100 * dm_minus_smooth[valid_atr] / atr_12h[valid_atr]
        
        # DX and ADX
        dx = np.full_like(di_plus, np.nan, dtype=np.float64)
        di_sum = di_plus + di_minus
        valid_di = ~np.isnan(di_sum) & (di_sum > 0)
        dx[valid_di] = 100 * np.abs(di_plus[valid_di] - di_minus[valid_di]) / di_sum[valid_di]
        
        adx_12h = np.full_like(dx, np.nan, dtype=np.float64)
        if len(dx) >= 27:  # Need 14+14 for ADX
            adx_12h[26] = np.nanmean(dx[14:28])
            for i in range(27, len(dx)):
                adx_12h[i] = adx_12h[i-1] * (1 - alpha) + alpha * dx[i]
    else:
        adx_12h = np.full(len(df_12h), np.nan)
    
    # Align HTF ADX to 6h timeframe
    if len(adx_12h) > 0:
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    else:
        adx_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 28)  # Donchian, Volume MA, ATR, ADX warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Trend filter: ADX > 25 indicates strong trend
        trend_strong = adx_12h_aligned[i] > 25
        
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with trend alignment
        breakout_long = (price >= high_roll[i]) and trend_strong and vol_confirm
        breakout_short = (price <= low_roll[i]) and trend_strong and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals