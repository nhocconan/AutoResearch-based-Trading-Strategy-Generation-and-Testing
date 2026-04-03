#!/usr/bin/env python3
"""
Experiment #1808: 12h Donchian(20) Breakout + 1w HMA Trend + Volume + ATR Stoploss
HYPOTHESIS: 12h Donchian breakouts aligned with 1w HMA trend and volume confirmation (>1.5x average) capture medium-term swings in both bull and bear markets. The 1w timeframe filters out noise from shorter-term fluctuations, while the 12h Donchian provides clear breakout levels. Position size fixed at 0.25 to balance return and drawdown. Target: 75-150 total trades over 4 years (19-37/year) by using tight entry conditions and multi-timeframe confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1808_12h_donchian20_1w_hma_vol_v1"
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
    # HMA(21): Hull Moving Average
    def hull_moving_average(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma
    hma_1w = hull_moving_average(close_1w, 21)
    trend_1w = np.where(close_1w > hma_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
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
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Require 1w trend alignment
        trend_following = trend_1w_aligned[i] != 0  # Should always be ±1
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if trend_following and volume_spike:
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