#!/usr/bin/env python3
"""
Experiment #2869: 4h Donchian(20) Breakout + HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: 4h Donchian(20) breakouts capture strong directional moves. HMA(21) from 1d timeframe filters 
for trend alignment, reducing false breakouts. Volume spike (>2x MA20) confirms institutional participation. 
ATR(14) stoploss limits drawdown. Primary timeframe 4h targets 75-200 trades over 4 years (19-50/year).
Works in both bull and bear markets by taking breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2869_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA(21) trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) on 1d close
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(arr, period):
        if period <= 0:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1, dtype=np.float64)
        return np.convolve(arr, weights[::-1], mode='full')[-len(arr):] / weights.sum()
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_1d = wma(raw_hma, sqrt_len)
    
    # Trend: 1 if close > HMA, -1 if close < HMA
    trend_1d = np.where(close_1d > hma_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    lookback = 20
    donch_upper = np.full(n, np.nan)
    donch_lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donch_upper[i] = np.max(high[i - lookback + 1:i + 1])
        donch_lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = max(50, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # ATR approximation from price range
                atr_estimate = (high[i] - low[i]) * 0.5
                # Stoploss: exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Take profit: exit if price reaches opposite Donchian level
                elif price <= donch_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                atr_estimate = (high[i] - low[i]) * 0.5
                # Stoploss: exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Take profit: exit if price reaches opposite Donchian level
                elif price >= donch_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Get 1d HMA trend bias
            trend_bias = trend_1d_aligned[i]
            
            # Long entry: price breaks above Donchian upper in HMA uptrend
            if trend_bias > 0 and price > donch_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower in HMA downtrend
            elif trend_bias < 0 and price < donch_lower[i]:
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