#!/usr/bin/env python3
"""
Experiment #547: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts on 6h timeframe aligned with weekly pivot-derived trend direction (from 1d HTF) and volume spikes capture strong momentum with controlled trade frequency. Weekly pivot levels (calculated from 1d data) provide structural trend filter that adapts to changing market regimes. Volume confirmation (>1.5x average) ensures participation. ATR-based stoploss (2.0) manages risk. Discrete position sizing (0.25) limits drawdown. Targets 50-150 total trades over 4 years by using tight entry conditions (breakout + pivot trend + volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_547_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from daily data
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    # We approximate weekly values using rolling window of 5 days (1 trading week)
    if len(close_1d) >= 5:
        # Rolling weekly high/low/close
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot point
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Weekly support/resistance levels
        weekly_r1 = 2 * weekly_pivot - weekly_low
        weekly_s1 = 2 * weekly_pivot - weekly_high
        weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
        weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
        weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
        weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
        
        # Trend determination: price above/below weekly pivot
        bullish_trend = close_1d > weekly_pivot
        bearish_trend = close_1d < weekly_pivot
    else:
        weekly_pivot = np.full(len(close_1d), np.nan)
        weekly_r1 = np.full(len(close_1d), np.nan)
        weekly_s1 = np.full(len(close_1d), np.nan)
        weekly_r2 = np.full(len(close_1d), np.nan)
        weekly_s2 = np.full(len(close_1d), np.nan)
        weekly_r3 = np.full(len(close_1d), np.nan)
        weekly_s3 = np.full(len(close_1d), np.nan)
        bullish_trend = np.zeros(len(close_1d), dtype=bool)
        bearish_trend = np.zeros(len(close_1d), dtype=bool)
    
    # Align weekly pivot and trend to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(np.float64))
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(np.float64))
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for weekly pivot calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Weekly Pivot Trend Filter ---
        # Bullish trend: price above weekly pivot
        bullish_trend = bullish_trend_aligned[i] > 0.5
        # Bearish trend: price below weekly pivot
        bearish_trend = bearish_trend_aligned[i] > 0.5
        
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
            
            # Optional: time-based exit after 8 bars (~2 days on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + bullish weekly pivot trend
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + bearish weekly pivot trend
            elif breakout_down and bearish_trend:
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