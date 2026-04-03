#!/usr/bin/env python3
"""
Experiment #1528: 12h Donchian(20) Breakout + 1w Trend + Volume Spike + Chop Filter
HYPOTHESIS: 12h Donchian breakouts with 1-week trend alignment, volume confirmation (>2.0x average), and choppiness regime filter (CHOP < 38.2 = trending) capture medium-term swings in both bull and bear markets. The 12h timeframe reduces trade frequency to avoid fee drag while allowing sufficient signals. Position size fixed at 0.25 to balance return and drawdown. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1528_12h_donchian20_1w_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    hma_1w = calculate_hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === HTF: 1d data for chop regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    tr1[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr1).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # +DM and -DM for 1d
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, and TR
    tr_period = 14
    atr_1d_smooth = pd.Series(tr1).ewm(span=tr_period, min_periods=tr_period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=tr_period, min_periods=tr_period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=tr_period, min_periods=tr_period, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / (atr_1d_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_1d_smooth + 1e-10)
    
    # DX and Chopiness Index (CHOP)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=tr_period, min_periods=tr_period, adjust=False).mean().values
    chop = 100 * np.log10(atr_1d_smooth.sum() / (np.max(high_1d) - np.min(low_1d)) + 1e-10) / np.log10(tr_period)
    # Fix: Calculate proper CHOP using rolling sum of TR and rolling range
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    high_max = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_min = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (high_max - low_min + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1w trend alignment
        trend_following = trend_1w_aligned[i] != 0
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        # Chop filter: only trade when market is trending (CHOP < 38.2)
        chop_trending = chop_aligned[i] < 38.2
        
        if trend_following and volume_spike and chop_trending:
            # Breakout: price breaks above upper band OR below lower band
            if price > donch_high[i] and trend_1w_aligned[i] > 0:  # Uptrend breakout
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and trend_1w_aligned[i] < 0:  # Downtrend breakdown
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

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    wma_half = np.zeros_like(close)
    for i in range(half_period, len(close)):
        wma_half[i] = np.dot(close[i-half_period+1:i+1], np.arange(1, half_period+1)) / (half_period * (half_period + 1) / 2)
    
    # WMA for full period
    wma_full = np.zeros_like(close)
    for i in range(period, len(close)):
        wma_full[i] = np.dot(close[i-period+1:i+1], np.arange(1, period+1)) / (period * (period + 1) / 2)
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw_hma with sqrt_period
    hma = np.zeros_like(close)
    for i in range(sqrt_period, len(close)):
        hma[i] = np.dot(raw_hma[i-sqrt_period+1:i+1], np.arange(1, sqrt_period+1)) / (sqrt_period * (sqrt_period + 1) / 2)
    
    return hma