#!/usr/bin/env python3
"""
Experiment #1518: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: Daily Donchian breakouts with weekly HMA(21) trend alignment and volume confirmation (>1.5x average) capture medium-term swings while reducing noise. Position size fixed at 0.25 to limit drawdown. Target: 75-150 total trades over 4 years (19-37/year) by using tight entry conditions on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1518_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
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
    # HMA(21) = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    def hma(arr, period):
        half = period // 2
        sqrt_n = int(np.sqrt(period))
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full_like(arr, np.nan)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_n)
    hma_1w_raw = hma(close_1w, 21)
    hma_1w = np.full_like(close_1w, np.nan)
    hma_1w[len(close_1w) - len(hma_1w_raw):] = hma_1w_raw
    trend_1w = np.where(close_1w > hma_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
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
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
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
        trend_following = trend_1w_aligned[i]
        
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