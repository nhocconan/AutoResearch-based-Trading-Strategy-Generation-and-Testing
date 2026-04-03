#!/usr/bin/env python3
"""
Experiment #256: 12h Donchian(20) breakout + 1d HMA trend + volume confirmation + chop filter
HYPOTHESIS: Donchian breakouts capture strong momentum. 1d HMA(21) defines trend regime: only take breakouts in trend direction. Volume confirmation (>1.5x average) ensures participation. Choppiness filter (CHOP > 61.8) avoids false breakouts in ranging markets. ATR stoploss (2.5x) manages risk. Discrete position sizing (0.25) limits drawdown. Works in bull via upward breakouts and in bear via downward breakouts, with regime filter preventing counter-trend entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_256_12h_donchian20_1d_regime_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA(21) trend regime ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return np.convolve(series, weights / weights.sum(), mode='valid')
    def hma(series, period):
        half = period // 2
        sqrt_n = int(np.sqrt(period))
        wma_half = wma(series, half)
        wma_full = wma(series, period)
        raw = 2 * wma_half - wma_full
        return wma(raw, sqrt_n)
    hma_21_1d = hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # === 12h Indicators: Donchian(20) channels ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: ATR(14) for stoploss and volume MA ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 12h Indicators: Choppiness Index (CHOP) regime filter ===
    def choppiness_index(high, low, close, period=14):
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = np.zeros_like(close)
        denom = hh - ll
        denom = np.where(denom == 0, 1e-10, denom)  # avoid div by zero
        chop = 100 * np.log10(tr_sum / denom) / np.log10(period)
        return chop
    chop = choppiness_index(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Regime Filters ---
        # Trend regime: price above/below 1d HMA(21)
        bullish_regime = price > hma_21_1d_aligned[i]
        bearish_regime = price < hma_21_1d_aligned[i]
        # Chop regime: avoid false breakouts in ranging markets (CHOP > 61.8 = choppy)
        chop_filter = chop[i] <= 61.8  # only allow when not choppy
        
        # --- Donchian Breakout Signals ---
        # Upper breakout: close > previous period's high
        upper_breakout = close[i] > highest_20[i-1] if i > 0 else False
        # Lower breakout: close < previous period's low
        lower_breakout = close[i] < lowest_20[i-1] if i > 0 else False
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite breakout or regime change
                if lower_breakout or (bearish_regime and volume_spike):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite breakout or regime change
                if upper_breakout or (bullish_regime and volume_spike):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if volume_spike and chop_filter:
            # Long entry: bullish breakout in bullish regime
            if upper_breakout and bullish_regime:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: bearish breakout in bearish regime
            elif lower_breakout and bearish_regime:
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