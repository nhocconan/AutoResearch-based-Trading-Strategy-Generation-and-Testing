#!/usr/bin/env python3
"""
Experiment #3139: 6h Donchian Breakout + 12h ADX Trend + Volume Spike
HYPOTHESIS: 6h Donchian(20) breakouts with 12h ADX(14) > 25 trend filter and volume confirmation 
captures strong momentum moves while filtering choppy markets. Works in bull markets via trend 
continuation and bear markets via mean reversion from extremes when ADX < 20 (range regime). 
Position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3139_6h_donchian20_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        # Smoothed values
        tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / tr_smooth
        minus_di = 100 * dm_minus_smooth / tr_smooth
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (np.abs(plus_di) + np.abs(minus_di)) * 100
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if ADX weakens (<20) suggesting range market
                elif adx_12h_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if ADX weakens (<20) suggesting range market
                elif adx_12h_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Only trade when ADX indicates trending market (>25)
            if adx_12h_aligned[i] > 25:
                # Long entry: price breaks above Donchian high
                if price > highest_high[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                # Short entry: price breaks below Donchian low
                elif price < lowest_low[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No trade in ranging markets
        else:
            signals[i] = 0.0
    
    return signals