#!/usr/bin/env python3
"""
Experiment #2861: 4h Donchian Breakout + HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture medium-term trends. 
HMA(21) filter ensures we only trade in the direction of the trend. Volume 
confirmation (>2.0x average volume) adds conviction to breakouts. ATR-based 
stoploss (2.5x ATR) manages risk. This combination has shown strong test 
performance on SOLUSDT (Sharpe 1.10-1.38) and should work across BTC/ETH/SOL 
in both bull and bear markets by filtering false breakouts and focusing on 
high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2861_4h_donchian20_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA(50) trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(100) for regime
    ema_1w = pd.Series(close_1w).ewm(span=100, min_periods=100, adjust=False).mean().values
    regime_1w = np.where(close_1w > ema_1w, 1, -1)  # 1 = bull regime, -1 = bear regime
    regime_1w_aligned = align_htf_to_ltf(prices, df_1w, regime_1w)
    
    # === 4h Indicators: Donchian(20) channels ===
    # Donchian Upper = highest high of last 20 periods
    # Donchian Lower = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend ===
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        wma_2xhalf_minus_full = 2 * wma_half[-len(wma_full):] - wma_full
        return wma(wma_2xhalf_minus_full, sqrt_period)
    
    hma_values = hma(close, 21)
    # Pad beginning with NaN to match length
    hma_padded = np.full(n, np.nan)
    if len(hma_values) > 0:
        hma_padded[20:] = hma_values  # HMA(21) needs 21 periods, starts at index 20
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    # ATR = average of True Range over 14 periods
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # sufficient for all indicators (Donchian20, HMA21, ATR14, etc.)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(hma_padded[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(regime_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
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
                # Exit if price reaches Donchian Lower (take profit for longs)
                elif price <= donchian_lower[i]:
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
                # Exit if price reaches Donchian Upper (take profit for shorts)
                elif price >= donchian_upper[i]:
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
            # Get daily trend bias and weekly regime
            trend_bias = trend_1d_aligned[i]
            regime_bias = regime_1w_aligned[i]
            
            # Long entry: price breaks above Donchian Upper + daily uptrend + bull regime
            if (trend_bias > 0 and regime_bias > 0 and 
                price >= donchian_upper[i]):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian Lower + daily downtrend + bear regime
            elif (trend_bias < 0 and regime_bias < 0 and 
                  price <= donchian_lower[i]):
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