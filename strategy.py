#!/usr/bin/env python3
"""
Experiment #1356: 12h Donchian(20) Breakout + 1d Trend + Volume Confirmation + Chop Filter
HYPOTHESIS: Donchian(20) breakouts on 12h timeframe with 1d trend filter and choppiness regime filter capture sustainable moves while avoiding whipsaws in ranging markets. Volume confirmation (>1.7x average) ensures institutional participation. Designed for low trade frequency (target: 75-150 total over 4 years) to minimize fee drag. ATR-based stoploss (2.5x) manages risk. Works in both bull (follow 1d uptrend breakouts) and bear (follow 1d downtrend breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1356_12h_donchian20_1d_vol_chop_v1"
timeframe = "12h"
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
    # 1d trend: EMA(50) slope
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_slope = np.zeros_like(ema_50)
    ema_slope[1:] = np.where(ema_50[1:] > ema_50[:-1], 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    # === HTF: 1d data for choppiness regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # True Range for 1d
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    # Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (log(n) * (max(high_n) - min(low_n))))
    # Simplified: use ATR ratio and range
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_1d = max_high_1d - min_low_1d
    chop = np.ones(len(close_1d)) * 50  # default neutral
    mask = range_1d > 0
    chop[mask] = 100 * np.log10(sum_atr_1d[mask] / (np.log(14) * range_1d[mask]))
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
            np.isnan(trend_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
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
        # Regime filter: only trade when not too choppy (CHOP < 61.8 = trending)
        not_choppy = chop_aligned[i] < 61.8
        # Volume confirmation: require volume spike (> 1.7x average)
        volume_spike = vol_ratio[i] > 1.7
        
        if not_choppy and volume_spike:
            # Breakout: price breaks above upper band OR below lower band
            if price > donch_high[i] and trend_1d_aligned[i] > 0:  # 1d uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and trend_1d_aligned[i] < 0:  # 1d downtrend
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