#!/usr/bin/env python3
"""
Experiment #4457: 4h Donchian(20) Breakout + 1d EMA Trend + Volume Confirmation + Chop Filter
HYPOTHESIS: 4h Donchian(20) breakouts aligned with 1d EMA50 trend and confirmed by volume (>1.5x average) 
and choppiness regime (CHOP < 61.8 = trending) capture institutional momentum with minimal false signals. 
The chop filter avoids ranging markets where breakouts fail. Targets 75-200 total trades over 4 years 
(19-50/year) with position size 0.25. Works in both bull (breakouts with trend) and bear (breakouts 
against trend filtered by chop/volume) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4457_4h_donchian20_1d_ema_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d EMA50 for trend bias ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        close_1d = pd.Series(df_1d['close'].values)
        ema_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === Precompute HTF: 1w Chopiness Index for regime filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 14:
        high_1w = pd.Series(df_1w['high'].values)
        low_1w = pd.Series(df_1w['low'].values)
        close_1w = pd.Series(df_1w['close'].values)
        # True Range
        tr1 = high_1w[1:] - low_1w[1:]
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        # Sum of ATR over 14 periods
        sum_atr_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
        # MaxHigh - MinLow over 14 periods
        max_high_1w = high_1w.rolling(window=14, min_periods=14).max().values
        min_low_1w = low_1w.rolling(window=14, min_periods=14).min().values
        range_1w = max_high_1w - min_low_1w
        # Chopiness Index: 100 * log10(sumATR / range) / log10(14)
        chop_1w = 100 * (np.log10(sum_atr_1w) - np.log10(range_1w)) / np.log10(14)
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14, 50, 14)  # Donchian, vol MA, ATR, EMA, chop
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # Chop filter: only trade when trending (CHOP < 61.8)
        trending_regime = chop_1w_aligned[i] < 61.8
        
        # 1d EMA bias: price > EMA = long bias, price < EMA = short bias
        long_bias = price > ema_1d_aligned[i]
        short_bias = price < ema_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout + long bias + volume + trending regime
        long_entry = breakout_up and long_bias and volume_confirm and trending_regime
        
        # Short conditions: downward breakout + short bias + volume + trending regime
        short_entry = breakout_down and short_bias and volume_confirm and trending_regime
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals