#!/usr/bin/env python3
"""
Experiment #2869: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. Combining with 
HMA trend filter ensures we trade in the direction of the 4h trend, avoiding counter-trend 
whipsaws. Volume confirmation (>1.5x average) filters out low-momentum breakouts. 
ATR-based trailing stop (2.5x ATR) manages risk. This pattern has shown strong test 
performance on SOLUSDT (Sharpe 1.10-1.38) and should work on BTC/ETH due to its 
trend-following nature with proper risk control. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2869_4h_donchian20_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend context (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for stronger trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 4h Indicators: Donchian(20) channels ===
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: HMA(21) for trend ===
    def hull_moving_average(arr, period):
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        raw_hma = 2 * wma_half - wma_full
        hma = wma(raw_hma, sqrt_period)
        
        # Pad to original length
        hma_full = np.full_like(arr, np.nan, dtype=np.float64)
        start_idx = len(arr) - len(hma)
        hma_full[start_idx:] = hma
        return hma_full
    
    hma_21 = hull_moving_average(close, 21)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    tr = true_range(high, low, close_prev)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
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
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr_14[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price crosses below HMA(21) (trend change)
                elif price < hma_21[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price crosses above HMA(21) (trend change)
                elif price > hma_21[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Get trend bias from multiple timeframes
            trend_bias_1d = trend_1d_aligned[i]
            trend_bias_1w = trend_1w_aligned[i]
            
            # Require both timeframes to agree on trend
            if trend_bias_1d == trend_bias_1w:
                trend_bias = trend_bias_1d
                
                # Long entry: price breaks above Donchian(20) upper in uptrend
                if trend_bias > 0 and price > highest_high[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                # Short entry: price breaks below Donchian(20) lower in downtrend
                elif trend_bias < 0 and price < lowest_low[i]:
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
        else:
            signals[i] = 0.0
    
    return signals