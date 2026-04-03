#!/usr/bin/env python3
"""
Experiment #261: 4h Donchian(20) Breakout + HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian breakouts capture momentum bursts. HMA(21) on 1d defines trend direction (bullish if rising, bearish if falling). Volume spike (>2.0x 20-bar MA) confirms institutional participation. ATR(14) stoploss (2.5x) manages risk. Discrete position sizing (0.30) limits drawdown. Works in bull markets via breakouts with trend and in bear markets via mean reversion at Donchian lows during low volatility (chop > 61.8). Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_261_4h_donchian20_hma_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA(21) trend and Chop regime ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # HMA(21) on 1d close: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        return pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values
    
    hma_1d = hma(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Chop regime on 1d: CHOP = 100 * log10(sum(ATR(14)) / (max(high)-min(low))) / log10(14)
    def chop(high_arr, low_arr, close_arr, period=14):
        tr = np.maximum(high_arr - low_arr, np.maximum(np.abs(high_arr - np.roll(close_arr, 1)), np.abs(low_arr - np.roll(close_arr, 1))))
        tr[0] = high_arr[0] - low_arr[0]
        atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
        hh = pd.Series(high_arr).rolling(window=period, min_periods=period).max()
        ll = pd.Series(low_arr).rolling(window=period, min_periods=period).min()
        chop_val = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        return chop_val.values
    
    chop_1d = chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    def donchian_channels(high_arr, low_arr, period=20):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max()
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min()
        return upper.values, lower.values
    
    donch_ub, donch_lb = donchian_channels(high, low, 20)
    
    # === 4h Indicators: ATR(14) for stoploss and volume thresholds ===
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # max(20, 21, 14, 20) + buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_ub[i]) or np.isnan(donch_lb[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Trend from HMA(21) on 1d: rising = bullish, falling = bearish ---
        hma_rising = hma_1d_aligned[i] > hma_1d_aligned[i-1]
        hma_falling = hma_1d_aligned[i] < hma_1d_aligned[i-1]
        
        # --- Chop regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (breakout) ---
        chop_value = chop_1d_aligned[i]
        chop_ranging = chop_value > 61.8
        chop_trending = chop_value < 38.2
        
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
                # Exit conditions: Donchian lower band touch during chop or HMA reversal
                if chop_ranging and low[i] <= donch_lb[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                if not hma_rising and hma_falling and volume_spike:
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
                # Exit conditions: Donchian upper band touch during chop or HMA reversal
                if chop_ranging and high[i] >= donch_ub[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                if not hma_falling and hma_rising and volume_spike:
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
        if volume_spike:
            # Donchian breakout entries
            if chop_trending:  # Trending regime: breakout in direction of trend
                if high[i] > donch_ub[i] and hma_rising:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif low[i] < donch_lb[i] and hma_falling:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            elif chop_ranging:  # Ranging regime: mean reversion at extremes
                if low[i] <= donch_lb[i]:  # Price at Donchian long -> mean reversion short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif high[i] >= donch_ub[i]:  # Price at Donchian short -> mean reversion long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals