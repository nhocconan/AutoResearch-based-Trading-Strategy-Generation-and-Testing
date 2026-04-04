#!/usr/bin/env python3
"""
Experiment #2312: 12h Donchian(20) Breakout + HMA(21) Trend + Volume Spike
HYPOTHESIS: 12h Donchian breakouts with 1d/1w HMA trend alignment and volume confirmation
capture significant moves while filtering noise. Works in bull markets (breakouts with trend)
and bear markets (mean reversion fails due to trend filter, reducing false signals).
Timeframe: 12h targets 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2312_12h_donchian20_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate HMA(21) on 1d
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        diff = 2 * wma2 - wma1
        return pd.Series(diff).ewm(span=sqrt, adjust=False).mean().values
    hma_1d = hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for HMA trend ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    hma_1w = hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 12h Indicators: Donchian(20), ATR(14), Volume MA(20) ===
    # Donchian channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume MA for spike detection
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
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks Donchian low (strong support break)
                elif price <= donch_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks Donchian high (strong resistance break)
                elif price >= donch_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require both 1d and 1w trend alignment for bias filter
        trend_bias_1d = trend_1d_aligned[i]
        trend_bias_1w = trend_1w_aligned[i]
        
        # Only trade when both timeframes agree
        if trend_bias_1d == trend_bias_1w:
            trend_bias = trend_bias_1d  # Either 1 or -1
        else:
            trend_bias = 0  # No clear trend, skip
        
        # Volume confirmation: require volume spike (> 2.0x average - strict to limit trades)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks Donchian high with uptrend on both timeframes
            if trend_bias > 0 and price >= donch_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks Donchian low with downtrend on both timeframes
            elif trend_bias < 0 and price <= donch_low[i]:
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