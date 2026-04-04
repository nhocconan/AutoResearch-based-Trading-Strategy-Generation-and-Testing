#!/usr/bin/env python3
"""
Experiment #2857: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation
HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. 
Combining with HMA(21) trend filter ensures we only trade in direction of 
medium-term trend. Volume confirmation (>1.5x average) filters false breakouts. 
4h timeframe provides optimal balance of signal quality and trade frequency 
(19-50 trades/year target). ATR-based stoploss manages risk. Works in both 
bull and bear markets by following the trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2857_4h_donchian20_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) on daily close
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        wma_2half_minus_full = 2 * wma_half - wma_full
        # Pad to original length
        pad = len(arr) - len(wma_2half_minus_full)
        wma_2half_minus_full_padded = np.concatenate([np.full(pad, np.nan), wma_2half_minus_full])
        return wma(wma_2half_minus_full_padded, sqrt)
    
    hma_1d = hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for additional trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA(50) on weekly close for stronger trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    
    warmup = max(lookback, 20, 14) + 5  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Check stoploss
            if position_side > 0:  # Long
                if price <= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price >= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Get trend biases (both must agree)
            trend_bias_1d = trend_1d_aligned[i]
            trend_bias_1w = trend_1w_aligned[i]
            
            # Long entry: price breaks above Donchian upper + both trends up
            if (trend_bias_1d > 0 and trend_bias_1w > 0 and 
                price > highest_high[i]):
                in_position = True
                position_side = 1
                entry_price = close[i]
                stoploss_price = entry_price - 2.0 * atr[i]  # 2*ATR stop
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower + both trends down
            elif (trend_bias_1d < 0 and trend_bias_1w < 0 and 
                  price < lowest_low[i]):
                in_position = True
                position_side = -1
                entry_price = close[i]
                stoploss_price = entry_price + 2.0 * atr[i]  # 2*ATR stop
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals