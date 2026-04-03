#!/usr/bin/env python3
"""
Experiment #1501: 4h Donchian(20) Breakout + 1d Trend + Volume Confirmation + Chop Filter
HYPOTHESIS: Donchian(20) breakouts on 4h capture medium-term swings with 1d EMA(50) trend filter for direction.
Volume confirmation (>1.3x average) and choppiness regime filter (CHOP > 38.2) reduce false breakouts.
ATR-based stoploss (2.0) manages risk. Designed for 19-50 trades/year (75-200 total over 4 years) by using
tight entry conditions and multi-timeframe confluence. Works in bull/bear markets by following 1d trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1501_4h_donchian20_1d_trend_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for chop regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    # True Range for 1w
    tr1 = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        tr1[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    tr1[0] = high_1w[0] - low_1w[0]
    atr1w = pd.Series(tr1).ewm(span=14, min_periods=14, adjust=False).mean().values
    # +DM and -DM for 1w
    up_move = np.zeros(len(high_1w))
    down_move = np.zeros(len(high_1w))
    for i in range(1, len(high_1w)):
        up_move[i] = high_1w[i] - high_1w[i-1]
        down_move[i] = low_1w[i-1] - low_1w[i]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed +DM, -DM, ATR
    tr_ma = pd.Series(atr1w).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_ma
    minus_di = 100 * minus_dm_smooth / tr_ma
    # DX and Choppiness
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    chop = 100 * np.log10(tr_ma * np.sqrt(14)) / np.log10(dx + 1e-10)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # === 4h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
            np.isnan(trend_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment
        trend_following = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        # Chop regime filter: require CHOP > 38.2 (trending market)
        chop_filter = chop_aligned[i] > 38.2
        
        if trend_following != 0 and volume_spike and chop_filter:
            # Breakout: price breaks above upper band OR below lower band
            if price > donch_high[i] and trend_following > 0:  # Uptrend breakout
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and trend_following < 0:  # Downtrend breakdown
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